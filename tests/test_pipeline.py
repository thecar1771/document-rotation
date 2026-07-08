from pathlib import Path

from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.validation import OcrRecognition


class FakeOrientationClient:
    def orientation_scores(self, image):
        scores = [
            AngleScore(90.0, 0.94),
            AngleScore(0.0, 0.80),
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
            if crop.width > crop.height:
                results.append([OcrRecognition("1234567890 \uc815\uc0c1", 0.95)])
            else:
                results.append([OcrRecognition("x", 0.20)])
        return results


def test_pipeline_uses_best_recognition_score_instead_of_model_thresholds(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "rotated.png"
    Image.new("RGB", (500, 100), "white").save(input_path)
    preprocessor = RotationPreprocessor(
        orientation_client=FakeOrientationClient(),
        crop_recognizer=FakeRecognizer(),
        config=RotationConfig(candidate_top_k=2, max_candidates=8),
    )

    result = preprocessor.process(input_path, output_path)

    assert result.output_path == output_path
    assert result.decision.angle == 0.0
    assert result.decision.should_rotate is False
    assert Image.open(output_path).size == (500, 100)
    assert result.trace.candidate_angles
    assert max(score.score for score in result.trace.validation_scores) > 0.0
