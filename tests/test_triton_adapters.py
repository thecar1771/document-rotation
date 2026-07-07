import numpy as np
from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.triton_client import ModelTensorNames, TritonCropRecognizer, TritonOrientationClient


class FakeTritonClient:
    def infer(self, model_name, inputs, outputs, timeout_ms):
        if model_name == "ocr_korean_rec":
            return {"logits": np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)}
        return {"probabilities": np.array([[0.01, 0.96, 0.02, 0.01]], dtype=np.float32)}

    def is_model_ready(self, model_name):
        return True


def test_triton_orientation_client_returns_three_model_scores():
    client = TritonOrientationClient(FakeTritonClient(), RotationConfig())

    scores = client.orientation_scores(Image.new("RGB", (512, 512), "white"))

    assert [score.model_name for score in scores] == ["deep_image", "paddle_doc_ori", "doctr_page"]
    assert all(score.best.angle == 90.0 for score in scores)


def test_triton_crop_recognizer_returns_one_result_per_crop():
    recognizer = TritonCropRecognizer(FakeTritonClient(), RotationConfig(), alphabet=["", "가"])

    results = recognizer.recognize([Image.new("RGB", (120, 32), "white")])

    assert len(results) == 1
    assert results[0][0].text
    assert results[0][0].confidence > 0.0


class IoAwareFakeTritonClient:
    def __init__(self):
        self.calls = []

    def infer(self, model_name, inputs, outputs, timeout_ms):
        self.calls.append((model_name, tuple(inputs.keys()), tuple(outputs)))
        if model_name == "ocr_korean_rec":
            return {outputs[0]: np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)}
        return {outputs[0]: np.array([[0.01, 0.96, 0.02, 0.01]], dtype=np.float32)}

    def is_model_ready(self, model_name):
        return True


def test_triton_adapters_use_model_io_mapping():
    fake = IoAwareFakeTritonClient()
    config = RotationConfig()
    model_io = {
        "orientation_deep_image": ModelTensorNames(input="input", output="output"),
        "orientation_paddle_doc_ori": ModelTensorNames(input="x", output="fetch_name_0"),
        "orientation_doctr_page": ModelTensorNames(input="input", output="output"),
        "ocr_korean_rec": ModelTensorNames(input="x", output="fetch_name_0"),
    }
    orientation = TritonOrientationClient(fake, config, model_io=model_io)
    recognizer = TritonCropRecognizer(fake, config, alphabet=["", "가"], model_io=model_io)

    orientation.orientation_scores(Image.new("RGB", (512, 512), "white"))
    recognizer.recognize([Image.new("RGB", (120, 32), "white")])

    assert ("orientation_paddle_doc_ori", ("x",), ("fetch_name_0",)) in fake.calls
    assert ("ocr_korean_rec", ("x",), ("fetch_name_0",)) in fake.calls


class ShapeAwareFakeTritonClient:
    def __init__(self):
        self.input_shapes = {}

    def infer(self, model_name, inputs, outputs, timeout_ms):
        self.input_shapes[model_name] = next(iter(inputs.values())).shape
        if model_name == "ocr_korean_rec":
            return {outputs[0]: np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)}
        return {outputs[0]: np.array([[0.01, 0.96, 0.02, 0.01]], dtype=np.float32)}

    def is_model_ready(self, model_name):
        return True


def test_triton_adapters_use_model_specific_input_shapes():
    fake = ShapeAwareFakeTritonClient()
    config = RotationConfig()
    model_io = {
        "orientation_deep_image": ModelTensorNames("input", "output", input_shape=[1, 3, 384, 384]),
        "orientation_paddle_doc_ori": ModelTensorNames("x", "fetch_name_0", input_shape=[1, 3, 224, 224]),
        "orientation_doctr_page": ModelTensorNames("input", "logits", input_shape=[1, 3, 512, 512]),
        "ocr_korean_rec": ModelTensorNames("x", "fetch_name_0", input_shape=[1, 3, 48, -1]),
    }

    orientation = TritonOrientationClient(fake, config, model_io=model_io)
    recognizer = TritonCropRecognizer(fake, config, alphabet=["", "가"], model_io=model_io)
    orientation.orientation_scores(Image.new("RGB", (900, 700), "white"))
    recognizer.recognize([Image.new("RGB", (120, 32), "white")])

    assert fake.input_shapes["orientation_deep_image"] == (1, 3, 384, 384)
    assert fake.input_shapes["orientation_paddle_doc_ori"] == (1, 3, 224, 224)
    assert fake.input_shapes["orientation_doctr_page"] == (1, 3, 512, 512)
    assert fake.input_shapes["ocr_korean_rec"][2] == 48


class LogitOrientationFakeTritonClient:
    def infer(self, model_name, inputs, outputs, timeout_ms):
        return {outputs[0]: np.array([[-1.0, 3.0, 0.0, 0.0]], dtype=np.float32)}

    def is_model_ready(self, model_name):
        return True


def test_orientation_client_normalizes_logits_to_probabilities():
    client = TritonOrientationClient(LogitOrientationFakeTritonClient(), RotationConfig())

    scores = client.orientation_scores(Image.new("RGB", (512, 512), "white"))

    assert scores[0].best.angle == 90.0
    assert 0.80 < scores[0].best.score < 1.0
