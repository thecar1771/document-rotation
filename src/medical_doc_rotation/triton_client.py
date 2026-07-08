from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.validation import OcrRecognition


@dataclass(frozen=True)
class ModelTensorNames:
    input: str
    output: str
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None


class TritonClientProtocol(Protocol):
    def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str],
        timeout_ms: int,
    ) -> dict[str, np.ndarray]:
        ...

    def is_model_ready(self, model_name: str) -> bool:
        ...


@dataclass
class TritonHttpClient:
    url: str

    def __post_init__(self) -> None:
        import tritonclient.http as httpclient

        self._client = httpclient.InferenceServerClient(url=self.url)

    def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        outputs: list[str],
        timeout_ms: int,
    ) -> dict[str, np.ndarray]:
        import tritonclient.http as httpclient

        triton_inputs = []
        for name, value in inputs.items():
            request_input = httpclient.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
            request_input.set_data_from_numpy(value)
            triton_inputs.append(request_input)
        triton_outputs = [httpclient.InferRequestedOutput(name) for name in outputs]
        result = self._client.infer(
            model_name,
            triton_inputs,
            outputs=triton_outputs,
            request_timeout=timeout_ms / 1000,
        )
        return {name: result.as_numpy(name) for name in outputs}

    def is_model_ready(self, model_name: str) -> bool:
        return bool(self._client.is_model_ready(model_name))


def np_to_triton_dtype(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    if dtype == np.uint8:
        return "UINT8"
    if dtype == np.int64:
        return "INT64"
    raise ValueError(f"Unsupported Triton dtype: {dtype}")


ANGLE_ORDER = [0.0, 90.0, 180.0, 270.0]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOCTR_PAGE_ORIENTATION_MEAN = np.array([0.694, 0.695, 0.693], dtype=np.float32)
DOCTR_PAGE_ORIENTATION_STD = np.array([0.299, 0.296, 0.301], dtype=np.float32)


def _image_to_nchw(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def _normalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)


def _array_to_nchw(array: np.ndarray) -> np.ndarray:
    return np.transpose(array.astype(np.float32), (2, 0, 1))[None, ...]


def _resize_shorter_side(image: Image.Image, shorter_side: int) -> Image.Image:
    width, height = image.size
    scale = shorter_side / max(1, min(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = size
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def _resize_with_symmetric_pad(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    target_width, target_height = size
    scale = min(target_width / max(1, width), target_height / max(1, height))
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.BILINEAR).convert("RGB")
    canvas = Image.new("RGB", size, "white")
    offset = ((target_width - resized.width) // 2, (target_height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def _preprocess_deep_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    shorter_side = max(size) + 32
    resized = _resize_shorter_side(image.convert("RGB"), shorter_side)
    cropped = _center_crop(resized, size)
    array = np.asarray(cropped).astype(np.float32) / 255.0
    return _array_to_nchw(_normalize(array, IMAGENET_MEAN, IMAGENET_STD))


def _preprocess_paddle_doc_ori(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    return _array_to_nchw(_normalize(array, IMAGENET_MEAN, IMAGENET_STD))


def _preprocess_doctr_page(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = _resize_with_symmetric_pad(image.convert("RGB"), size)
    array = np.asarray(resized).astype(np.float32) / 255.0
    return _array_to_nchw(_normalize(array, DOCTR_PAGE_ORIENTATION_MEAN, DOCTR_PAGE_ORIENTATION_STD))


def _preprocess_orientation_image(logical_name: str, image: Image.Image, tensor_names: ModelTensorNames) -> np.ndarray:
    if logical_name == "deep_image":
        return _preprocess_deep_image(image, _image_size_from_shape(tensor_names, (384, 384)))
    if logical_name == "paddle_doc_ori":
        return _preprocess_paddle_doc_ori(image, _image_size_from_shape(tensor_names, (224, 224)))
    if logical_name == "doctr_page":
        return _preprocess_doctr_page(image, _image_size_from_shape(tensor_names, (512, 512)))
    return _image_to_nchw(image, _image_size_from_shape(tensor_names, (512, 512)))


def _image_size_from_shape(
    tensor_names: ModelTensorNames,
    default: tuple[int, int],
) -> tuple[int, int]:
    if not tensor_names.input_shape:
        return default
    dims = tensor_names.input_shape
    if len(dims) >= 4 and dims[0] in {-1, 0, 1}:
        dims = dims[1:]
    if len(dims) >= 3 and dims[0] in {1, 3}:
        height = dims[1] if dims[1] > 0 else default[1]
        width = dims[2] if dims[2] > 0 else default[0]
        return (width, height)
    if len(dims) >= 2:
        height = dims[-2] if dims[-2] > 0 else default[1]
        width = dims[-1] if dims[-1] > 0 else default[0]
        return (width, height)
    return default


def _input_width_is_dynamic(tensor_names: ModelTensorNames) -> bool:
    if not tensor_names.input_shape:
        return False
    dims = tensor_names.input_shape
    if len(dims) >= 4 and dims[0] in {-1, 0, 1}:
        dims = dims[1:]
    return len(dims) >= 3 and dims[2] <= 0


def _image_to_chw(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


class TritonOrientationClient:
    def __init__(
        self,
        client: TritonClientProtocol,
        config: RotationConfig,
        model_io: dict[str, ModelTensorNames] | None = None,
    ):
        self.client = client
        self.config = config
        self.model_io = model_io or {}

    def orientation_scores(self, image: Image.Image) -> list[OrientationScores]:
        model_map = {
            "deep_image": self.config.model_names.deep_image,
            "paddle_doc_ori": self.config.model_names.paddle_doc_ori,
            "doctr_page": self.config.model_names.doctr_page,
        }
        results: list[OrientationScores] = []
        for logical_name, model_name in model_map.items():
            tensor_names = self.model_io.get(model_name, ModelTensorNames(input="input", output="probabilities"))
            payload = _preprocess_orientation_image(logical_name, image, tensor_names)
            response = self.client.infer(
                model_name=model_name,
                inputs={tensor_names.input: payload},
                outputs=[tensor_names.output],
                timeout_ms=self.config.ensemble_timeout_ms,
            )
            probabilities = _as_probabilities(response[tensor_names.output][0])
            results.append(
                OrientationScores(
                    model_name=logical_name,
                    scores=[
                        AngleScore(angle=angle, score=float(probabilities[index]))
                        for index, angle in enumerate(ANGLE_ORDER)
                    ],
                )
            )
        return results


class TritonCropRecognizer:
    def __init__(
        self,
        client: TritonClientProtocol,
        config: RotationConfig,
        alphabet: list[str],
        model_io: dict[str, ModelTensorNames] | None = None,
    ):
        self.client = client
        self.config = config
        self.alphabet = alphabet
        self.model_io = model_io or {}

    def recognize(self, crops: list[Image.Image]) -> list[list[OcrRecognition]]:
        if not crops:
            return []
        tensor_names = self.model_io.get(
            self.config.model_names.korean_rec,
            ModelTensorNames(input="input", output="logits"),
        )
        batch = self._prepare_ocr_batch(crops, tensor_names)
        response = self.client.infer(
            model_name=self.config.model_names.korean_rec,
            inputs={tensor_names.input: batch},
            outputs=[tensor_names.output],
            timeout_ms=self.config.validation_timeout_ms,
        )
        logits = response[tensor_names.output]
        return [self._decode(row) for row in logits]

    def _prepare_ocr_batch(self, crops: list[Image.Image], tensor_names: ModelTensorNames) -> np.ndarray:
        size = _image_size_from_shape(tensor_names, (160, 32))
        target_height = size[1]
        if not _input_width_is_dynamic(tensor_names):
            return np.stack([_image_to_chw(crop, size) for crop in crops], axis=0)

        max_width = max(8, self.config.ocr_max_width)
        resized: list[np.ndarray] = []
        widths: list[int] = []
        for crop in crops:
            scale = target_height / max(1, crop.height)
            width = min(max_width, max(8, round(crop.width * scale)))
            resized_crop = _image_to_chw(crop, (width, target_height))
            resized.append(resized_crop)
            widths.append(width)

        batch_width = min(max_width, max(widths))
        batch = np.ones((len(resized), 3, target_height, batch_width), dtype=np.float32)
        for index, item in enumerate(resized):
            width = min(item.shape[-1], batch_width)
            batch[index, :, :, :width] = item[:, :, :width]
        return batch

    def _decode(self, logits: np.ndarray) -> list[OcrRecognition]:
        indices = np.argmax(logits, axis=-1)
        confidences = np.max(_softmax(logits), axis=-1)
        blank_index = len(self.alphabet) - 1
        chars: list[str] = []
        kept_confidences: list[float] = []
        previous_index: int | None = None
        for index, confidence in zip(indices, confidences):
            current_index = int(index)
            if current_index == previous_index:
                continue
            previous_index = current_index
            if current_index == blank_index:
                continue
            if 0 <= current_index < len(self.alphabet) and self.alphabet[current_index]:
                chars.append(self.alphabet[current_index])
                kept_confidences.append(float(confidence))
        text = "".join(chars)
        confidence = float(np.mean(kept_confidences)) if kept_confidences else 0.0
        return [OcrRecognition(text=text, confidence=confidence)]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _as_probabilities(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if np.any(array < 0) or not np.isclose(float(np.sum(array)), 1.0, atol=1e-3):
        return _softmax(array)
    return array
