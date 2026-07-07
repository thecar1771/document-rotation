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


def _image_to_nchw(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


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
            payload = _image_to_nchw(image, _image_size_from_shape(tensor_names, (512, 512)))
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
        size = _image_size_from_shape(tensor_names, (160, 32))
        batch = np.concatenate([_image_to_nchw(crop, size) for crop in crops], axis=0)
        response = self.client.infer(
            model_name=self.config.model_names.korean_rec,
            inputs={tensor_names.input: batch},
            outputs=[tensor_names.output],
            timeout_ms=self.config.validation_timeout_ms,
        )
        logits = response[tensor_names.output]
        return [self._decode(row) for row in logits]

    def _decode(self, logits: np.ndarray) -> list[OcrRecognition]:
        indices = np.argmax(logits, axis=-1)
        confidences = np.max(_softmax(logits), axis=-1)
        chars = [
            self.alphabet[index]
            for index in indices
            if 0 <= index < len(self.alphabet) and self.alphabet[index]
        ]
        text = "".join(chars)
        confidence = float(np.mean(confidences)) if len(confidences) else 0.0
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
