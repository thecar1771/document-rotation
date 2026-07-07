from pathlib import Path

from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.validation import OcrRecognition


class FakeOrientationClient:
    def orientation_scores(self, image):
        scores = [
            AngleScore(0.0, 0.02),
            AngleScore(90.0, 0.94),
            AngleScore(180.0, 0.02),
            AngleScore(270.0, 0.02),
        ]
        return [
            OrientationScores("deep_image", scores),
            OrientationScores("paddle_doc_ori", scores),
            OrientationScores("doctr_page", scores),
        ]


class FakeRecognizer:
    def recognize(self, crops):
        results = []
        for crop in crops:
            if crop.height > crop.width:
                results.append([OcrRecognition("진료비 12,000", 0.95)])
            else:
                results.append([OcrRecognition("x", 0.20)])
        return results


def test_pipeline_rotates_and_saves_image(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "rotated.png"
    Image.new("RGB", (500, 100), "white").save(input_path)
    preprocessor = RotationPreprocessor(
        orientation_client=FakeOrientationClient(),
        crop_recognizer=FakeRecognizer(),
        config=RotationConfig(),
    )

    result = preprocessor.process(input_path, output_path)

    assert result.output_path == output_path
    assert result.decision.should_rotate is True
    assert Image.open(output_path).size == (100, 500)
