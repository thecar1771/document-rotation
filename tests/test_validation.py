from PIL import Image

from medical_doc_rotation.types import AngleCandidate
from medical_doc_rotation.validation import OcrRecognition, score_candidate_crops


class FakeRecognizer:
    def recognize(self, crops):
        return [
            [OcrRecognition(text="\uc544\ubb34\ub2e8\uc5b4 12,000", confidence=0.91)],
            [OcrRecognition(text="\ud14c\uc2a4\ud2b8 34,000", confidence=0.88)],
        ][: len(crops)]


def test_score_candidate_crops_uses_recognition_quality_without_anchor_patterns():
    image = Image.new("RGB", (400, 300), "white")
    candidate = AngleCandidate(angle=90.0, reason="unit")

    scores = score_candidate_crops(
        image=image,
        candidates=[candidate],
        recognizer=FakeRecognizer(),
        crops_per_candidate=2,
    )

    assert len(scores) == 1
    assert scores[0].angle == 90.0
    assert scores[0].avg_confidence > 0.8
    assert scores[0].recognized_chars >= 8
    assert scores[0].score > 1.0
