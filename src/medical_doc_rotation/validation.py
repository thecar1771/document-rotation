import re
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from medical_doc_rotation.geometry import select_evidence_crops
from medical_doc_rotation.image_ops import rotate_image
from medical_doc_rotation.types import AngleCandidate, ValidationScore

ANCHOR_PATTERN = re.compile(r"(진료비|본인부담|납입|영수증|세부내역|합계)")
NUMERIC_PATTERN = re.compile(r"(\d{1,3}(,\d{3})+|\d{4}[.-]\d{1,2}[.-]\d{1,2}|\d{3}-\d{2}-\d{5})")


@dataclass(frozen=True)
class OcrRecognition:
    text: str
    confidence: float


class CropRecognizer(Protocol):
    def recognize(self, crops: list[Image.Image]) -> list[list[OcrRecognition]]:
        ...


def score_candidate_crops(
    image: Image.Image,
    candidates: list[AngleCandidate],
    recognizer: CropRecognizer,
    crops_per_candidate: int,
) -> list[ValidationScore]:
    scores: list[ValidationScore] = []
    for candidate in candidates:
        rotated = rotate_image(image, candidate.angle)
        boxes = select_evidence_crops(rotated, count=crops_per_candidate)
        crops = [rotated.crop(box) for box in boxes]
        recognitions = recognizer.recognize(crops)
        flat = [item for crop_result in recognitions for item in crop_result]
        if not flat:
            scores.append(ValidationScore(candidate.angle, 0.0, 0.0, 0.0, 0, 1.0))
            continue
        avg_conf = sum(item.confidence for item in flat) / len(flat)
        text = " ".join(item.text for item in flat)
        recognized_chars = sum(1 for char in text if char.isdigit() or "\uac00" <= char <= "\ud7a3")
        recognized_ratio = recognized_chars / max(1, len(text))
        pattern_hits = len(ANCHOR_PATTERN.findall(text)) + len(NUMERIC_PATTERN.findall(text))
        broken_penalty = sum(1 for token in text.split() if len(token) == 1) / max(1, len(text.split()))
        score = avg_conf + recognized_ratio + pattern_hits * 0.25 - broken_penalty * 0.5
        scores.append(
            ValidationScore(
                angle=candidate.angle,
                score=score,
                avg_confidence=avg_conf,
                recognized_ratio=recognized_ratio,
                pattern_hits=pattern_hits,
                broken_token_penalty=broken_penalty,
            )
        )
    return scores
