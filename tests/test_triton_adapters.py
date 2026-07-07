import numpy as np
from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.triton_client import TritonCropRecognizer, TritonOrientationClient


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
