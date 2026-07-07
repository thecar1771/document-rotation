from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.types import AngleScore, OrientationScores


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
