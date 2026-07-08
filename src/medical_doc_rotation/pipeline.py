import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.decision import build_recognition_candidates, decide_by_recognition
from medical_doc_rotation.geometry import estimate_fine_angle
from medical_doc_rotation.image_ops import load_image, resize_for_working, rotate_image, save_image
from medical_doc_rotation.types import OrientationScores, RotationResult, RotationTrace
from medical_doc_rotation.validation import CropRecognizer, score_candidate_crops


class OrientationClient(Protocol):
    def orientation_scores(self, image: Image.Image) -> list[OrientationScores]:
        ...


@dataclass
class RotationPreprocessor:
    orientation_client: OrientationClient
    crop_recognizer: CropRecognizer
    config: RotationConfig

    def process(self, input_path: Path | str, output_path: Path | str) -> RotationResult:
        start = time.perf_counter()
        source_path = Path(input_path)
        target_path = Path(output_path)
        original = load_image(source_path)
        working = resize_for_working(original, self.config.max_working_long_edge)
        fine_angle = estimate_fine_angle(working)
        model_scores = self.orientation_client.orientation_scores(working)
        candidates = build_recognition_candidates(model_scores, fine_angle, self.config)
        validation_scores = score_candidate_crops(
            image=working,
            candidates=candidates,
            recognizer=self.crop_recognizer,
            crops_per_candidate=self.config.crops_per_candidate,
        )
        decision = decide_by_recognition(validation_scores, self.config)
        output_image = rotate_image(original, decision.angle) if decision.should_rotate else original.copy()
        save_image(output_image, target_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        trace = RotationTrace(
            model_scores=model_scores,
            candidate_angles=candidates,
            validation_scores=validation_scores,
        )
        return RotationResult(source_path, target_path, decision, elapsed_ms, trace)
