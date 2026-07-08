from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.types import (
    AngleCandidate,
    AngleScore,
    FineAngleEstimate,
    OrientationScores,
    RotationDecision,
    ValidationScore,
)

COARSE_ANGLES = [0.0, 90.0, 180.0, 270.0]


def normalize_angle(angle: float) -> float:
    normalized = angle % 360.0
    if abs(normalized - 360.0) < 1e-6:
        return 0.0
    return normalized


def angular_distance(left: float, right: float) -> float:
    distance = abs(normalize_angle(left) - normalize_angle(right))
    return min(distance, 360.0 - distance)


def fuse_orientation_scores(model_scores: list[OrientationScores], weights: dict[str, float]) -> OrientationScores:
    totals = {angle: 0.0 for angle in COARSE_ANGLES}
    weight_total = 0.0
    for item in model_scores:
        weight = weights.get(item.model_name, 1.0)
        weight_total += weight
        for score in item.scores:
            angle = normalize_angle(score.angle)
            totals[angle] = totals.get(angle, 0.0) + score.score * weight
    divisor = weight_total or 1.0
    return OrientationScores(
        model_name="fused",
        scores=[AngleScore(angle=angle, score=value / divisor) for angle, value in totals.items()],
    )


def _apply_fine_angle(coarse_angle: float, fine_angle: FineAngleEstimate, config: RotationConfig) -> float:
    if fine_angle.confidence >= config.fine_angle_min_confidence:
        return normalize_angle(coarse_angle + fine_angle.angle)
    return normalize_angle(coarse_angle)


def select_candidates(
    fused: OrientationScores,
    model_scores: list[OrientationScores],
    fine_angle: FineAngleEstimate,
    config: RotationConfig,
) -> list[AngleCandidate]:
    angles: list[float] = [fused.best.angle]
    model_best_angles = [score.best.angle for score in model_scores]
    if 90.0 in model_best_angles and 270.0 in model_best_angles:
        angles.extend([90.0, 270.0])
    if any(score.best.angle == 0.0 and score.best.score >= config.strong_zero_score for score in model_scores):
        angles.append(0.0)
    for score in fused.ordered[1 : config.max_candidates]:
        if len({round(item) % 360 for item in angles}) >= config.normal_candidate_count:
            break
        angles.append(score.angle)

    unique: list[AngleCandidate] = []
    seen: set[int] = set()
    for angle in angles:
        final_angle = _apply_fine_angle(angle, fine_angle, config)
        key = round(final_angle) % 360
        if key not in seen and len(unique) < config.max_candidates:
            seen.add(key)
            unique.append(AngleCandidate(angle=final_angle, reason="candidate"))
    return unique


def _agreement_count(model_scores: list[OrientationScores], angle: float) -> int:
    target = round(normalize_angle(angle)) % 360
    return sum(1 for score in model_scores if round(normalize_angle(score.best.angle)) % 360 == target)


def _validation_support(angle: float, validation_scores: list[ValidationScore], config: RotationConfig) -> bool:
    if not validation_scores:
        return True
    ordered = sorted(validation_scores, key=lambda item: item.score, reverse=True)
    best = ordered[0]
    if angular_distance(best.angle, angle) > 5.0:
        return False
    if best.score < config.validation_min_score:
        return False
    if len(ordered) == 1:
        return True
    return best.score - ordered[1].score >= config.validation_min_margin


def _zero_is_strong(model_scores: list[OrientationScores], config: RotationConfig) -> bool:
    return any(score.best.angle == 0.0 and score.best.score >= config.strong_zero_score for score in model_scores)


def decide_final_angle(
    fused: OrientationScores,
    model_scores: list[OrientationScores],
    fine_angle: FineAngleEstimate,
    validation_scores: list[ValidationScore],
    config: RotationConfig,
) -> RotationDecision:
    best = fused.best
    second = fused.second_best
    if best.angle == 0.0:
        return RotationDecision(angle=0.0, should_rotate=False, reason="fused_zero")
    if _zero_is_strong(model_scores, config):
        return RotationDecision(angle=0.0, should_rotate=False, reason="strong_zero")
    if best.score < config.min_ensemble_score:
        return RotationDecision(angle=0.0, should_rotate=False, reason="low_ensemble_score")
    if best.score - second.score < config.min_score_margin:
        return RotationDecision(angle=0.0, should_rotate=False, reason="low_margin")
    if _agreement_count(model_scores, best.angle) < 2:
        return RotationDecision(angle=0.0, should_rotate=False, reason="low_agreement")
    final_angle = _apply_fine_angle(best.angle, fine_angle, config)
    if not _validation_support(final_angle, validation_scores, config):
        return RotationDecision(angle=0.0, should_rotate=False, reason="validation_reject")
    return RotationDecision(angle=final_angle, should_rotate=True, reason="accepted")
