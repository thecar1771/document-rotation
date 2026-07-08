from dataclasses import dataclass, field


@dataclass(frozen=True)
class TritonModelNames:
    deep_image: str = "orientation_deep_image"
    paddle_doc_ori: str = "orientation_paddle_doc_ori"
    doctr_page: str = "orientation_doctr_page"
    korean_rec: str = "ocr_korean_rec"


@dataclass(frozen=True)
class RotationConfig:
    default_angle: float = 0.0
    max_working_long_edge: int = 1600
    min_working_long_edge: int = 1024
    min_ensemble_score: float = 0.85
    min_score_margin: float = 0.20
    strong_zero_score: float = 0.88
    fine_angle_min_confidence: float = 0.55
    max_candidates: int = 3
    normal_candidate_count: int = 2
    crops_per_candidate: int = 10
    validation_min_score: float = 0.50
    validation_min_margin: float = 0.15
    ocr_max_width: int = 640
    ensemble_timeout_ms: int = 250
    validation_timeout_ms: int = 350
    model_names: TritonModelNames = field(default_factory=TritonModelNames)
