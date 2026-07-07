# Medical Document Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python medical-document rotation preprocessor with Triton-served orientation/OCR models, conservative angle decisions, model download/setup scripts, and tests.

**Architecture:** The package loads an image, creates a bounded working image, estimates fine angle locally with OpenCV, calls Triton for coarse orientation models and OCR-recognition validation, chooses a conservative final correction angle, rotates the original image, and saves it for the existing downstream process. The model setup script prepares a Triton model repository but does not start Triton because the server already exists.

**Tech Stack:** Python 3.11+, Pillow, OpenCV, NumPy, tritonclient, Hugging Face Hub, pytest, ONNX models served through NVIDIA Triton.

---

## File Structure

- Create: `pyproject.toml` - package metadata, dependencies, console entrypoints, pytest config.
- Create: `src/medical_doc_rotation/__init__.py` - public package exports.
- Create: `src/medical_doc_rotation/config.py` - dataclass configuration for thresholds, Triton names, and model setup.
- Create: `src/medical_doc_rotation/types.py` - shared dataclasses for angles, model scores, validation scores, and results.
- Create: `src/medical_doc_rotation/image_ops.py` - image loading, EXIF normalization, resize, crop, and rotation.
- Create: `src/medical_doc_rotation/geometry.py` - OpenCV fine-angle estimation and deterministic evidence crop selection.
- Create: `src/medical_doc_rotation/decision.py` - ensemble fusion, candidate selection, and final angle decision rules.
- Create: `src/medical_doc_rotation/triton_client.py` - small Triton HTTP client wrapper and test double boundary.
- Create: `src/medical_doc_rotation/validation.py` - OCR recognition-only crop validation scoring.
- Create: `src/medical_doc_rotation/pipeline.py` - end-to-end rotation preprocessor orchestration.
- Create: `src/medical_doc_rotation/cli.py` - command line entrypoints for processing one image and checking Triton readiness.
- Create: `scripts/setup_models.py` - downloads model artifacts and writes Triton model repository configs.
- Create: `tests/conftest.py` - test fixtures and synthetic image helpers.
- Create: `tests/test_image_ops.py` - image resize and rotation tests.
- Create: `tests/test_geometry.py` - fine-angle and deterministic crop tests.
- Create: `tests/test_decision.py` - ensemble, candidate, and no-op protection tests.
- Create: `tests/test_validation.py` - OCR scoring tests with fake recognizer outputs.
- Create: `tests/test_pipeline.py` - full pipeline tests using fake Triton responses.
- Create: `tests/test_setup_models.py` - model repository generation tests without network downloads.

## Model Repository Targets

The setup script creates this structure under a user-provided `--repo-dir`:

```text
orientation_deep_image/1/model.onnx
orientation_deep_image/config.pbtxt
orientation_paddle_doc_ori/1/model.onnx
orientation_paddle_doc_ori/config.pbtxt
orientation_doctr_page/1/model.onnx
orientation_doctr_page/config.pbtxt
ocr_korean_rec/1/model.onnx
ocr_korean_rec/config.pbtxt
ocr_korean_rec/dict.txt
```

Initial external model sources:

- `DuarteBarbosa/deep-image-orientation-detection`, file `orientation_model_v2_0.9882.onnx`, MIT license accepted as a permissive exception because this model already exists in the target environment.
- `monkt/paddleocr-onnx`, document orientation model selected by repository file scan under `preprocessing/doc-orientation/`.
- `Felix92/onnxtr-mobilenet-v3-small-page-orientation`, file `model.onnx`, Apache-2.0 docTR/OnnxTR page-orientation artifact.
- `monkt/paddleocr-onnx`, Korean recognition model selected from `languages/korean/rec.onnx` and `languages/korean/dict.txt`.

## Task 1: Project Skeleton And Shared Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/medical_doc_rotation/__init__.py`
- Create: `src/medical_doc_rotation/config.py`
- Create: `src/medical_doc_rotation/types.py`
- Test: `tests/test_decision.py`

- [ ] **Step 1: Write failing tests for threshold defaults and score dataclasses**

Create `tests/test_decision.py` with:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_decision.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'medical_doc_rotation'`.

- [ ] **Step 3: Add package metadata and shared contracts**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "medical-doc-rotation"
version = "0.1.0"
description = "Medical document image rotation preprocessor with Triton model serving"
requires-python = ">=3.11"
dependencies = [
  "huggingface_hub>=0.23",
  "numpy>=1.24",
  "opencv-python-headless>=4.9",
  "Pillow>=10.0",
  "tritonclient[http]>=2.45",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[project.scripts]
medical-doc-rotate = "medical_doc_rotation.cli:main"
medical-doc-setup-models = "scripts.setup_models:main"

[tool.setuptools.packages.find]
where = ["src", "."]
include = ["medical_doc_rotation*", "scripts*"]

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
```

Create `src/medical_doc_rotation/__init__.py`:

```python
from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor

__all__ = ["RotationConfig", "RotationPreprocessor"]
```

Create `src/medical_doc_rotation/config.py`:

```python
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
    ensemble_timeout_ms: int = 250
    validation_timeout_ms: int = 350
    model_names: TritonModelNames = field(default_factory=TritonModelNames)
```

Create `src/medical_doc_rotation/types.py`:

```python
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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_decision.py -v
```

Expected: PASS for both tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/medical_doc_rotation tests/test_decision.py
git commit -m "feat: add rotation package contracts"
```

## Task 2: Image Loading, Resize, And Rotation

**Files:**
- Create: `src/medical_doc_rotation/image_ops.py`
- Test: `tests/test_image_ops.py`

- [ ] **Step 1: Write failing image operation tests**

Create `tests/test_image_ops.py` with:

```python
from pathlib import Path

from PIL import Image

from medical_doc_rotation.image_ops import load_image, resize_for_working, rotate_image, save_image


def test_resize_for_working_preserves_aspect_ratio():
    image = Image.new("RGB", (4000, 2000), "white")

    resized = resize_for_working(image, max_long_edge=1600)

    assert resized.size == (1600, 800)


def test_rotate_right_angle_swaps_width_and_height():
    image = Image.new("RGB", (500, 100), "white")

    rotated = rotate_image(image, 90.0)

    assert rotated.size == (100, 500)


def test_rotate_arbitrary_angle_expands_output():
    image = Image.new("RGB", (100, 100), "white")

    rotated = rotate_image(image, 45.0)

    assert rotated.width > 100
    assert rotated.height > 100


def test_save_and_load_round_trip(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.new("RGB", (20, 10), "white").save(input_path)

    loaded = load_image(input_path)
    save_image(loaded, output_path)

    assert Image.open(output_path).size == (20, 10)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_image_ops.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing functions from `medical_doc_rotation.image_ops`.

- [ ] **Step 3: Implement image operations**

Create `src/medical_doc_rotation/image_ops.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def save_image(image: Image.Image, path: Path | str, quality: int = 95) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=quality, subsampling=0)
    else:
        image.save(output_path)
    return output_path


def resize_for_working(image: Image.Image, max_long_edge: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image.copy()
    scale = max_long_edge / long_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def rotate_image(image: Image.Image, correction_angle: float, fill_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    normalized = correction_angle % 360
    if normalized == 0:
        return image.copy()
    if normalized == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if normalized == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if normalized == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image.rotate(correction_angle, expand=True, fillcolor=fill_color, resample=Image.Resampling.BICUBIC)


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_image_ops.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/image_ops.py tests/test_image_ops.py
git commit -m "feat: add image rotation operations"
```

## Task 3: OpenCV Fine-Angle Estimation And Evidence Crops

**Files:**
- Create: `src/medical_doc_rotation/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Create `tests/test_geometry.py` with:

```python
from PIL import Image, ImageDraw

from medical_doc_rotation.geometry import estimate_fine_angle, select_evidence_crops


def make_line_image(angle: float) -> Image.Image:
    image = Image.new("RGB", (400, 300), "white")
    canvas = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(canvas)
    for y in range(80, 230, 40):
        draw.line((60, y, 340, y), fill="black", width=3)
    return canvas.rotate(angle, expand=False, fillcolor="white")


def test_estimate_fine_angle_returns_correction_for_slanted_lines():
    image = make_line_image(7.0)

    estimate = estimate_fine_angle(image)

    assert estimate.confidence > 0.2
    assert -12.0 < estimate.angle < -2.0


def test_select_evidence_crops_is_deterministic_and_bounded():
    image = Image.new("RGB", (500, 300), "white")

    first = select_evidence_crops(image, count=6)
    second = select_evidence_crops(image, count=6)

    assert first == second
    assert len(first) == 6
    for left, top, right, bottom in first:
        assert 0 <= left < right <= 500
        assert 0 <= top < bottom <= 300
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_geometry.py -v
```

Expected: FAIL with missing `geometry` module.

- [ ] **Step 3: Implement geometry helpers**

Create `src/medical_doc_rotation/geometry.py`:

```python
import math

import cv2
import numpy as np
from PIL import Image

from medical_doc_rotation.image_ops import pil_to_rgb_array
from medical_doc_rotation.types import FineAngleEstimate


def _normalize_line_angle(angle: float) -> float:
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def estimate_fine_angle(image: Image.Image) -> FineAngleEstimate:
    rgb = pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=max(30, image.width // 8), maxLineGap=12)
    if lines is None:
        return FineAngleEstimate(angle=0.0, confidence=0.0)

    weighted_angles: list[tuple[float, float]] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue
        raw_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        normalized = _normalize_line_angle(raw_angle)
        if abs(normalized) <= 30:
            weighted_angles.append((normalized, length))

    if not weighted_angles:
        return FineAngleEstimate(angle=0.0, confidence=0.0)

    total_weight = sum(weight for _, weight in weighted_angles)
    mean_angle = sum(angle * weight for angle, weight in weighted_angles) / total_weight
    variance = sum(weight * (angle - mean_angle) ** 2 for angle, weight in weighted_angles) / total_weight
    confidence = min(1.0, len(weighted_angles) / 24.0) * max(0.0, 1.0 - min(1.0, math.sqrt(variance) / 15.0))
    return FineAngleEstimate(angle=-mean_angle, confidence=confidence)


def select_evidence_crops(image: Image.Image, count: int) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    crop_w = max(32, round(width * 0.42))
    crop_h = max(32, round(height * 0.18))
    anchors = [
        (0.29, 0.18), (0.71, 0.18),
        (0.29, 0.38), (0.71, 0.38),
        (0.29, 0.58), (0.71, 0.58),
        (0.29, 0.78), (0.71, 0.78),
        (0.50, 0.50), (0.75, 0.82),
        (0.50, 0.18), (0.50, 0.82),
    ]
    boxes: list[tuple[int, int, int, int]] = []
    for cx_ratio, cy_ratio in anchors[:count]:
        cx = round(width * cx_ratio)
        cy = round(height * cy_ratio)
        left = min(max(0, cx - crop_w // 2), max(0, width - crop_w))
        top = min(max(0, cy - crop_h // 2), max(0, height - crop_h))
        boxes.append((left, top, min(width, left + crop_w), min(height, top + crop_h)))
    return boxes
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_geometry.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/geometry.py tests/test_geometry.py
git commit -m "feat: estimate fine document angle"
```

## Task 4: Ensemble Fusion, Candidate Selection, And Final Decision

**Files:**
- Create: `src/medical_doc_rotation/decision.py`
- Modify: `tests/test_decision.py`

- [ ] **Step 1: Extend failing decision tests**

Append to `tests/test_decision.py`:

```python
from medical_doc_rotation.decision import (
    decide_final_angle,
    fuse_orientation_scores,
    select_candidates,
)
from medical_doc_rotation.types import FineAngleEstimate, ValidationScore


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_decision.py -v
```

Expected: FAIL with missing `medical_doc_rotation.decision`.

- [ ] **Step 3: Implement decision rules**

Create `src/medical_doc_rotation/decision.py`:

```python
from collections import Counter

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


def fuse_orientation_scores(model_scores: list[OrientationScores], weights: dict[str, float]) -> OrientationScores:
    totals = {angle: 0.0 for angle in COARSE_ANGLES}
    weight_total = 0.0
    for item in model_scores:
        weight = weights.get(item.model_name, 1.0)
        weight_total += weight
        for score in item.scores:
            totals[normalize_angle(score.angle)] = totals.get(normalize_angle(score.angle), 0.0) + score.score * weight
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
    for score in fused.ordered[1:config.max_candidates]:
        if len(set(angles)) >= config.normal_candidate_count:
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


def _validation_support(angle: float, validation_scores: list[ValidationScore]) -> bool:
    if not validation_scores:
        return True
    ordered = sorted(validation_scores, key=lambda item: item.score, reverse=True)
    best = ordered[0]
    if round(normalize_angle(best.angle)) % 360 != round(normalize_angle(angle)) % 360:
        return False
    if len(ordered) == 1:
        return best.score > 0.5
    return best.score - ordered[1].score >= 0.15


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
    if not _validation_support(final_angle, validation_scores):
        return RotationDecision(angle=0.0, should_rotate=False, reason="validation_reject")
    return RotationDecision(angle=final_angle, should_rotate=True, reason="accepted")
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_decision.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/decision.py tests/test_decision.py
git commit -m "feat: add conservative rotation decisions"
```

## Task 5: Triton Client Boundary And OCR Validation Scorer

**Files:**
- Create: `src/medical_doc_rotation/triton_client.py`
- Create: `src/medical_doc_rotation/validation.py`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_validation.py` with:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_validation.py -v
```

Expected: FAIL with missing `validation` module.

- [ ] **Step 3: Implement Triton client boundary and validation scorer**

Create `src/medical_doc_rotation/triton_client.py`:

```python
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TritonClientProtocol(Protocol):
    def infer(self, model_name: str, inputs: dict[str, np.ndarray], outputs: list[str], timeout_ms: int) -> dict[str, np.ndarray]:
        ...

    def is_model_ready(self, model_name: str) -> bool:
        ...


@dataclass
class TritonHttpClient:
    url: str

    def __post_init__(self) -> None:
        import tritonclient.http as httpclient

        self._client = httpclient.InferenceServerClient(url=self.url)

    def infer(self, model_name: str, inputs: dict[str, np.ndarray], outputs: list[str], timeout_ms: int) -> dict[str, np.ndarray]:
        import tritonclient.http as httpclient

        triton_inputs = []
        for name, value in inputs.items():
            request_input = httpclient.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
            request_input.set_data_from_numpy(value)
            triton_inputs.append(request_input)
        triton_outputs = [httpclient.InferRequestedOutput(name) for name in outputs]
        result = self._client.infer(model_name, triton_inputs, outputs=triton_outputs, request_timeout=timeout_ms / 1000)
        return {name: result.as_numpy(name) for name in outputs}

    def is_model_ready(self, model_name: str) -> bool:
        return bool(self._client.is_model_ready(model_name))


def np_to_triton_dtype(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    if dtype == np.uint8:
        return "UINT8"
    if dtype == np.int64:
        return "INT64"
    raise ValueError(f"Unsupported Triton dtype: {dtype}")
```

Create `src/medical_doc_rotation/validation.py`:

```python
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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_validation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/triton_client.py src/medical_doc_rotation/validation.py tests/test_validation.py
git commit -m "feat: score OCR validation crops"
```

## Task 6: Pipeline And CLI

**Files:**
- Create: `src/medical_doc_rotation/pipeline.py`
- Create: `src/medical_doc_rotation/cli.py`
- Modify: `src/medical_doc_rotation/__init__.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_pipeline.py` with:

```python
from pathlib import Path

from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.validation import OcrRecognition


class FakeOrientationClient:
    def orientation_scores(self, image):
        scores = [
            AngleScore(0.0, 0.02),
            AngleScore(90.0, 0.94),
            AngleScore(180.0, 0.02),
            AngleScore(270.0, 0.02),
        ]
        return [
            OrientationScores("deep_image", scores),
            OrientationScores("paddle_doc_ori", scores),
            OrientationScores("doctr_page", scores),
        ]


class FakeRecognizer:
    def recognize(self, crops):
        return [[OcrRecognition("진료비 12,000", 0.95)] for _ in crops]


def test_pipeline_rotates_and_saves_image(tmp_path: Path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "rotated.png"
    Image.new("RGB", (500, 100), "white").save(input_path)
    preprocessor = RotationPreprocessor(
        orientation_client=FakeOrientationClient(),
        crop_recognizer=FakeRecognizer(),
        config=RotationConfig(),
    )

    result = preprocessor.process(input_path, output_path)

    assert result.output_path == output_path
    assert result.decision.should_rotate is True
    assert Image.open(output_path).size == (100, 500)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: FAIL with missing `pipeline` module.

- [ ] **Step 3: Implement pipeline and CLI**

Create `src/medical_doc_rotation/pipeline.py`:

```python
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.decision import decide_final_angle, fuse_orientation_scores, select_candidates
from medical_doc_rotation.geometry import estimate_fine_angle
from medical_doc_rotation.image_ops import load_image, resize_for_working, rotate_image, save_image
from medical_doc_rotation.types import OrientationScores, RotationResult
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
        fused = fuse_orientation_scores(
            model_scores,
            {
                "doctr_page": 0.40,
                "deep_image": 0.35,
                "paddle_doc_ori": 0.25,
                "fused": 1.0,
            },
        )
        candidates = select_candidates(fused, model_scores, fine_angle, self.config)
        validation_scores = score_candidate_crops(
            image=working,
            candidates=candidates,
            recognizer=self.crop_recognizer,
            crops_per_candidate=self.config.crops_per_candidate,
        )
        decision = decide_final_angle(fused, model_scores, fine_angle, validation_scores, self.config)
        output_image = rotate_image(original, decision.angle) if decision.should_rotate else original.copy()
        save_image(output_image, target_path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RotationResult(source_path, target_path, decision, elapsed_ms)
```

Create `src/medical_doc_rotation/cli.py`:

```python
import argparse
from pathlib import Path

from medical_doc_rotation.config import RotationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate a medical document image before downstream processing.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--triton-url", default="localhost:8000")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(
        "CLI wiring requires Triton adapters from Task 7 before processing real images. "
        f"Received input={args.input}, output={args.output}, triton_url={args.triton_url}, config={RotationConfig()}."
    )
```

Modify `src/medical_doc_rotation/__init__.py`:

```python
from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor

__all__ = ["RotationConfig", "RotationPreprocessor"]
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/pipeline.py src/medical_doc_rotation/cli.py src/medical_doc_rotation/__init__.py tests/test_pipeline.py
git commit -m "feat: add rotation preprocessing pipeline"
```

## Task 7: Triton Model Adapters

**Files:**
- Modify: `src/medical_doc_rotation/triton_client.py`
- Modify: `src/medical_doc_rotation/validation.py`
- Create: `tests/test_triton_adapters.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/test_triton_adapters.py` with:

```python
import numpy as np
from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.triton_client import TritonCropRecognizer, TritonOrientationClient


class FakeTritonClient:
    def infer(self, model_name, inputs, outputs, timeout_ms):
        if model_name == "ocr_korean_rec":
            return {"logits": np.array([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32)}
        return {"probabilities": np.array([[0.01, 0.96, 0.02, 0.01]], dtype=np.float32)}

    def is_model_ready(self, model_name):
        return True


def test_triton_orientation_client_returns_three_model_scores():
    client = TritonOrientationClient(FakeTritonClient(), RotationConfig())

    scores = client.orientation_scores(Image.new("RGB", (512, 512), "white"))

    assert [score.model_name for score in scores] == ["deep_image", "paddle_doc_ori", "doctr_page"]
    assert all(score.best.angle == 90.0 for score in scores)


def test_triton_crop_recognizer_returns_one_result_per_crop():
    recognizer = TritonCropRecognizer(FakeTritonClient(), RotationConfig(), alphabet=["", "가"])

    results = recognizer.recognize([Image.new("RGB", (120, 32), "white")])

    assert len(results) == 1
    assert results[0][0].text
    assert results[0][0].confidence > 0.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_triton_adapters.py -v
```

Expected: FAIL with missing adapter classes.

- [ ] **Step 3: Implement Triton adapters**

Append to `src/medical_doc_rotation/triton_client.py`:

```python
from PIL import Image

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.image_ops import resize_for_working
from medical_doc_rotation.types import AngleScore, OrientationScores
from medical_doc_rotation.validation import OcrRecognition

ANGLE_ORDER = [0.0, 90.0, 180.0, 270.0]


def _image_to_nchw(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    resized = image.resize(size, Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized).astype(np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


class TritonOrientationClient:
    def __init__(self, client: TritonClientProtocol, config: RotationConfig):
        self.client = client
        self.config = config

    def orientation_scores(self, image: Image.Image) -> list[OrientationScores]:
        model_map = {
            "deep_image": self.config.model_names.deep_image,
            "paddle_doc_ori": self.config.model_names.paddle_doc_ori,
            "doctr_page": self.config.model_names.doctr_page,
        }
        payload = _image_to_nchw(image, (512, 512))
        results: list[OrientationScores] = []
        for logical_name, model_name in model_map.items():
            response = self.client.infer(
                model_name=model_name,
                inputs={"input": payload},
                outputs=["probabilities"],
                timeout_ms=self.config.ensemble_timeout_ms,
            )
            probabilities = response["probabilities"][0]
            results.append(
                OrientationScores(
                    model_name=logical_name,
                    scores=[AngleScore(angle=angle, score=float(probabilities[index])) for index, angle in enumerate(ANGLE_ORDER)],
                )
            )
        return results


class TritonCropRecognizer:
    def __init__(self, client: TritonClientProtocol, config: RotationConfig, alphabet: list[str]):
        self.client = client
        self.config = config
        self.alphabet = alphabet

    def recognize(self, crops: list[Image.Image]) -> list[list[OcrRecognition]]:
        if not crops:
            return []
        batch = np.concatenate([_image_to_nchw(crop, (160, 32)) for crop in crops], axis=0)
        response = self.client.infer(
            model_name=self.config.model_names.korean_rec,
            inputs={"input": batch},
            outputs=["logits"],
            timeout_ms=self.config.validation_timeout_ms,
        )
        logits = response["logits"]
        return [self._decode(row) for row in logits]

    def _decode(self, logits: np.ndarray) -> list[OcrRecognition]:
        indices = np.argmax(logits, axis=-1)
        confidences = np.max(_softmax(logits), axis=-1)
        chars = [self.alphabet[index] for index in indices if 0 <= index < len(self.alphabet) and self.alphabet[index]]
        text = "".join(chars)
        confidence = float(np.mean(confidences)) if len(confidences) else 0.0
        return [OcrRecognition(text=text, confidence=confidence)]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_triton_adapters.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/medical_doc_rotation/triton_client.py tests/test_triton_adapters.py
git commit -m "feat: add Triton inference adapters"
```

## Task 8: Model Download And Triton Repository Setup

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/setup_models.py`
- Test: `tests/test_setup_models.py`

- [ ] **Step 1: Write failing model setup tests**

Create `tests/test_setup_models.py` with:

```python
from pathlib import Path

from scripts.setup_models import ModelArtifact, write_model_config, write_repository_manifest


def test_write_model_config_creates_onnxruntime_config(tmp_path: Path):
    model_dir = tmp_path / "orientation_deep_image"

    write_model_config(model_dir, name="orientation_deep_image", input_shape=[1, 3, 512, 512], output_name="probabilities")

    text = (model_dir / "config.pbtxt").read_text(encoding="utf-8")
    assert 'name: "orientation_deep_image"' in text
    assert 'backend: "onnxruntime"' in text
    assert 'name: "input"' in text
    assert 'name: "probabilities"' in text


def test_manifest_records_sources(tmp_path: Path):
    artifact = ModelArtifact("unit", "repo/name", "model.onnx", "Apache-2.0")

    write_repository_manifest(tmp_path, [artifact])

    text = (tmp_path / "MODEL_SOURCES.md").read_text(encoding="utf-8")
    assert "repo/name" in text
    assert "Apache-2.0" in text
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/test_setup_models.py -v
```

Expected: FAIL with missing `scripts.setup_models`.

- [ ] **Step 3: Implement setup script**

Create `scripts/__init__.py`:

```python
"""Utility scripts for medical document rotation."""
```

Create `scripts/setup_models.py`:

```python
import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    repo_id: str
    filename: str
    license_name: str


ARTIFACTS = [
    ModelArtifact("orientation_deep_image", "DuarteBarbosa/deep-image-orientation-detection", "orientation_model_v2_0.9882.onnx", "MIT"),
    ModelArtifact("orientation_doctr_page", "Felix92/onnxtr-mobilenet-v3-small-page-orientation", "model.onnx", "Apache-2.0"),
    ModelArtifact("orientation_paddle_doc_ori", "monkt/paddleocr-onnx", "preprocessing/doc-orientation/model.onnx", "Apache-2.0"),
    ModelArtifact("ocr_korean_rec", "monkt/paddleocr-onnx", "languages/korean/rec.onnx", "Apache-2.0"),
]


def resolve_repo_file(repo_id: str, requested: str) -> str:
    files = list_repo_files(repo_id)
    if requested in files:
        return requested
    basename = Path(requested).name
    matches = [item for item in files if item.endswith(basename)]
    if matches:
        return matches[0]
    if "doc-orientation" in requested:
        matches = [item for item in files if "doc-orientation" in item and item.endswith(".onnx")]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve {requested} in {repo_id}")


def download_artifact(artifact: ModelArtifact, cache_dir: Path) -> Path:
    filename = resolve_repo_file(artifact.repo_id, artifact.filename)
    downloaded = hf_hub_download(repo_id=artifact.repo_id, filename=filename, cache_dir=cache_dir)
    return Path(downloaded)


def write_model_config(model_dir: Path, name: str, input_shape: list[int], output_name: str) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    dims = ", ".join(str(item) for item in input_shape[1:])
    text = f'''name: "{name}"
backend: "onnxruntime"
max_batch_size: 8
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ {dims} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
'''
    (model_dir / "config.pbtxt").write_text(text, encoding="utf-8")


def write_repository_manifest(repo_dir: Path, artifacts: list[ModelArtifact]) -> None:
    lines = ["# Model Sources", ""]
    for artifact in artifacts:
        lines.append(f"- `{artifact.name}`: `{artifact.repo_id}` / `{artifact.filename}` / `{artifact.license_name}`")
    (repo_dir / "MODEL_SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_models(repo_dir: Path, cache_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        source = download_artifact(artifact, cache_dir)
        version_dir = repo_dir / artifact.name / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, version_dir / "model.onnx")
        if artifact.name == "ocr_korean_rec":
            dict_source = Path(hf_hub_download(artifact.repo_id, "languages/korean/dict.txt", cache_dir=cache_dir))
            shutil.copy2(dict_source, repo_dir / artifact.name / "dict.txt")
            write_model_config(repo_dir / artifact.name, artifact.name, [1, 3, 32, 160], "logits")
        else:
            write_model_config(repo_dir / artifact.name, artifact.name, [1, 3, 512, 512], "probabilities")
    write_repository_manifest(repo_dir, ARTIFACTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download rotation models and prepare a Triton model repository.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".model-cache"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    install_models(args.repo_dir, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
pytest tests/test_setup_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts tests/test_setup_models.py
git commit -m "feat: add Triton model setup script"
```

## Task 9: Wire Real CLI To Triton Adapters

**Files:**
- Modify: `src/medical_doc_rotation/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI parser test**

Create `tests/test_cli.py` with:

```python
from pathlib import Path

from medical_doc_rotation.cli import build_parser


def test_cli_parser_accepts_required_paths_and_triton_url():
    parser = build_parser()

    args = parser.parse_args(["input.jpg", "output.jpg", "--triton-url", "localhost:8000"])

    assert args.input == Path("input.jpg")
    assert args.output == Path("output.jpg")
    assert args.triton_url == "localhost:8000"
```

- [ ] **Step 2: Run tests and verify current parser passes before wiring**

Run:

```bash
pytest tests/test_cli.py -v
```

Expected: PASS. This confirms parser stability before replacing the guarded CLI body.

- [ ] **Step 3: Replace CLI body with real Triton wiring**

Modify `src/medical_doc_rotation/cli.py`:

```python
import argparse
from pathlib import Path

from medical_doc_rotation.config import RotationConfig
from medical_doc_rotation.pipeline import RotationPreprocessor
from medical_doc_rotation.triton_client import TritonCropRecognizer, TritonHttpClient, TritonOrientationClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate a medical document image before downstream processing.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--dict-path", type=Path, required=True)
    return parser


def read_alphabet(path: Path) -> list[str]:
    values = [""] + [line.strip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    return values


def main() -> int:
    args = build_parser().parse_args()
    config = RotationConfig()
    client = TritonHttpClient(args.triton_url)
    orientation_client = TritonOrientationClient(client, config)
    recognizer = TritonCropRecognizer(client, config, read_alphabet(args.dict_path))
    preprocessor = RotationPreprocessor(orientation_client, recognizer, config)
    result = preprocessor.process(args.input, args.output)
    print(f"output={result.output_path} angle={result.decision.angle:.2f} rotate={result.decision.should_rotate} elapsed_ms={result.elapsed_ms:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update CLI test for required dictionary path**

Replace `tests/test_cli.py` with:

```python
from pathlib import Path

from medical_doc_rotation.cli import build_parser, read_alphabet


def test_cli_parser_accepts_required_paths_and_triton_url():
    parser = build_parser()

    args = parser.parse_args(["input.jpg", "output.jpg", "--triton-url", "localhost:8000", "--dict-path", "dict.txt"])

    assert args.input == Path("input.jpg")
    assert args.output == Path("output.jpg")
    assert args.triton_url == "localhost:8000"
    assert args.dict_path == Path("dict.txt")


def test_read_alphabet_adds_blank_token(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_text("가\n나\n", encoding="utf-8")

    assert read_alphabet(path) == ["", "가", "나"]
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
pytest tests/test_cli.py tests/test_pipeline.py tests/test_triton_adapters.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/medical_doc_rotation/cli.py tests/test_cli.py
git commit -m "feat: wire CLI to Triton adapters"
```

## Task 10: Full Test Run, Model Setup Dry Run, And Documentation

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/plans/2026-07-07-document-rotation-implementation.md`

- [ ] **Step 1: Run full unit test suite**

Run:

```bash
pytest -v
```

Expected: PASS.

- [ ] **Step 2: Run model setup tests without network**

Run:

```bash
pytest tests/test_setup_models.py -v
```

Expected: PASS.

- [ ] **Step 3: Create README with exact setup commands**

Create `README.md`:

```markdown
# Medical Document Rotation

Python preprocessor for medical document image rotation. It estimates a conservative correction angle, rotates the original image, saves the result, and passes that saved image to an existing downstream process.

## Install

```bash
pip install -e ".[test]"
```

## Prepare Triton Models

```bash
medical-doc-setup-models --repo-dir ./triton_model_repository --cache-dir ./.model-cache
```

Point the existing Triton server at `./triton_model_repository` or copy the generated model directories into the server's configured model repository.

## Run One Image

```bash
medical-doc-rotate input.jpg output.jpg --triton-url localhost:8000 --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt
```

## Safety Policy

The default decision is no rotation. A non-zero rotation is applied only when coarse orientation models, margin thresholds, agreement rules, and OCR crop validation support it.
```

- [ ] **Step 4: Run README command syntax smoke test**

Run:

```bash
python -m medical_doc_rotation.cli --help
python -m scripts.setup_models --help
```

Expected: both commands print help text and exit with code 0.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/plans/2026-07-07-document-rotation-implementation.md
git commit -m "docs: add rotation setup instructions"
```

## Self-Review Checklist

- Spec coverage: The plan implements image loading, EXIF normalization, working-image resize, OpenCV fine-angle estimation, three-model Triton coarse orientation, Top-K candidates, OCR recognition-only crop validation, conservative no-op decisions, original-image rotation, saved output, model download, Triton repository configs, tests, and README commands.
- Incomplete-marker scan: No task uses undefined future work; each code-changing step includes concrete code or command text.
- Type consistency: `RotationConfig`, `OrientationScores`, `FineAngleEstimate`, `AngleCandidate`, `ValidationScore`, `RotationDecision`, and `RotationResult` are introduced before dependent tasks use them.
