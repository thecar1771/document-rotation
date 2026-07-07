from PIL import Image

from medical_doc_rotation.types import AngleCandidate
from medical_doc_rotation.validation import OcrRecognition, score_candidate_crops


class FakeRecognizer:
    def recognize(self, crops):
        return [
            [OcrRecognition(text="진료비 12,000", confidence=0.91)],
            [OcrRecognition(text="합계 12,000", confidence=0.88)],
        ][: len(crops)]


def test_score_candidate_crops_rewards_medical_patterns():
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
    assert scores[0].pattern_hits >= 2
    assert scores[0].score > 1.0
