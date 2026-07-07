from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AngleScore:
    angle: float
    score: float


@dataclass(frozen=True)
class OrientationScores:
    model_name: str
    scores: list[AngleScore]

    @property
    def ordered(self) -> list[AngleScore]:
        return sorted(self.scores, key=lambda item: item.score, reverse=True)

    @property
    def best(self) -> AngleScore:
        return self.ordered[0]

    @property
    def second_best(self) -> AngleScore:
        ordered = self.ordered
        if len(ordered) < 2:
            return AngleScore(angle=self.best.angle, score=0.0)
        return ordered[1]


@dataclass(frozen=True)
class FineAngleEstimate:
    angle: float
    confidence: float


@dataclass(frozen=True)
class AngleCandidate:
    angle: float
    reason: str


@dataclass(frozen=True)
class ValidationScore:
    angle: float
    score: float
    avg_confidence: float
    recognized_ratio: float
    pattern_hits: int
    broken_token_penalty: float


@dataclass(frozen=True)
class RotationDecision:
    angle: float
    should_rotate: bool
    reason: str


@dataclass(frozen=True)
class RotationResult:
    input_path: Path
    output_path: Path
    decision: RotationDecision
    elapsed_ms: float
