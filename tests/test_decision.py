from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.decision import (
    decide_final_angle,
    fuse_orientation_scores,
    select_candidates,
)
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.types import FineAngleEstimate, ValidationScore


def test_default_thresholds_are_conservative():
    config = RotationConfig()

    assert config.min_ensemble_score == 0.85
    assert config.min_score_margin == 0.20
    assert config.max_candidates == 3
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


def test_select_candidates_keeps_90_and_270_conflict():
    candidates = select_candidates(
        fused=make_scores("fused", 90.0, 0.60),
        model_scores=[make_scores("a", 90.0), make_scores("b", 270.0), make_scores("c", 90.0)],
        fine_angle=FineAngleEstimate(angle=-2.0, confidence=0.8),
        config=RotationConfig(),
    )

    angles = {round(candidate.angle) % 360 for candidate in candidates}
    assert 88 in angles or 90 in angles
    assert 268 in angles or 270 in angles


def test_decider_keeps_zero_when_score_is_ambiguous():
    decision = decide_final_angle(
        fused=make_scores("fused", 90.0, 0.60),
        model_scores=[make_scores("a", 90.0), make_scores("b", 270.0), make_scores("c", 90.0)],
        fine_angle=FineAngleEstimate(angle=0.0, confidence=0.0),
        validation_scores=[],
        config=RotationConfig(),
    )

    assert decision.angle == 0.0
    assert decision.should_rotate is False


def test_decider_accepts_supported_high_confidence_non_zero():
    validation = [ValidationScore(90.0, 1.4, 0.9, 0.8, 3, 0.0)]
    decision = decide_final_angle(
        fused=make_scores("fused", 90.0, 0.94),
        model_scores=[make_scores("a", 90.0), make_scores("b", 90.0), make_scores("c", 0.0, 0.30)],
        fine_angle=FineAngleEstimate(angle=-2.0, confidence=0.8),
        validation_scores=validation,
        config=RotationConfig(),
    )

    assert round(decision.angle, 1) == 88.0
    assert decision.should_rotate is True
