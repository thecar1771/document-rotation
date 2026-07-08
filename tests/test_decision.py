from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.decision import (
    build_recognition_candidates,
    decide_by_recognition,
    fuse_orientation_scores,
)
from medical_doc_rotation.types import AngleScore, FineAngleEstimate, OrientationScores, ValidationScore


def test_default_thresholds_are_conservative():
    config = RotationConfig()

    assert config.min_ensemble_score == 0.85
    assert config.min_score_margin == 0.20
    assert config.max_candidates == 12
    assert config.candidate_top_k == 2
    assert config.default_angle == 0.0


def test_orientation_scores_report_best_and_second_best():
    scores = OrientationScores(
        model_name="unit",
        scores=[
            AngleScore(angle=0.0, score=0.1),
            AngleScore(angle=90.0, score=0.7),
            AngleScore(angle=270.0, score=0.2),
        ],
    )

    assert scores.best.angle == 90.0
    assert scores.second_best.angle == 270.0


def make_scores(name: str, best_angle: float, best_score: float = 0.92):
    other_score = (1.0 - best_score) / 3.0
    return OrientationScores(
        model_name=name,
        scores=[
            AngleScore(0.0, best_score if best_angle == 0.0 else other_score),
            AngleScore(90.0, best_score if best_angle == 90.0 else other_score),
            AngleScore(180.0, best_score if best_angle == 180.0 else other_score),
            AngleScore(270.0, best_score if best_angle == 270.0 else other_score),
        ],
    )


def test_fuse_orientation_scores_uses_weights():
    fused = fuse_orientation_scores(
        [make_scores("a", 90.0), make_scores("b", 90.0), make_scores("c", 0.0, 0.51)],
        weights={"a": 0.4, "b": 0.4, "c": 0.2},
    )

    assert fused.best.angle == 90.0
    assert fused.best.score > 0.70


def test_build_recognition_candidates_adds_inverse_angles_and_deduplicates():
    candidates = build_recognition_candidates(
        model_scores=[
            OrientationScores("deep_image", [AngleScore(90.0, 0.91), AngleScore(0.0, 0.05)]),
            OrientationScores("paddle_doc_ori", [AngleScore(88.0, 0.87), AngleScore(270.0, 0.12)]),
            OrientationScores("doctr_page", [AngleScore(195.0, 0.77)]),
        ],
        fine_angle=FineAngleEstimate(angle=4.0, confidence=0.9),
        config=RotationConfig(candidate_top_k=1, candidate_dedupe_degrees=5.0, max_candidates=12),
    )

    angles = [round(candidate.angle) % 360 for candidate in candidates]

    assert angles == [90, 270, 195, 165, 4, 356]


def test_decide_by_recognition_selects_highest_ocr_score_even_for_zero():
    decision = decide_by_recognition(
        validation_scores=[
            ValidationScore(90.0, 1.2, 0.8, 0.6, 12, 0.1),
            ValidationScore(0.0, 1.6, 0.9, 0.8, 20, 0.0),
        ],
        config=RotationConfig(),
    )

    assert decision.angle == 0.0
    assert decision.should_rotate is False
    assert decision.reason == "recognition_best"


def test_decide_by_recognition_selects_highest_ocr_score_for_non_zero():
    decision = decide_by_recognition(
        validation_scores=[
            ValidationScore(90.0, 2.2, 0.8, 0.6, 12, 0.1),
            ValidationScore(270.0, 1.6, 0.9, 0.8, 20, 0.0),
        ],
        config=RotationConfig(),
    )

    assert decision.angle == 90.0
    assert decision.should_rotate is True
    assert decision.reason == "recognition_best"
