# Recognition Candidate Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace threshold-heavy orientation decisions with candidate generation plus OCR recognition scoring, and expose traceable model/candidate logs.

**Architecture:** Orientation models only propose angles. Each proposed angle and its inverse are deduplicated into candidate rotations, then every candidate is scored by Korean OCR recognition crops. The final angle is the candidate with the highest OCR score.

**Tech Stack:** Python 3.10, Pillow, OpenCV, NumPy, Triton HTTP client, ONNX models served by existing Triton.

## Global Constraints

- Keep Triton model serving structure intact.
- Keep right-angle and arbitrary-angle rotation support.
- Do not use document-type anchor keywords in OCR validation.
- Prefer simple OCR-recognition score selection over hard 0/90/180/270 decision rules.
- Print model outputs, candidates, candidate OCR scores, and final angle for debugging.

---

### Task 1: Candidate Generation

**Files:**
- Modify: `src/medical_doc_rotation/decision.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Produces: `build_recognition_candidates(model_scores, fine_angle, config) -> list[AngleCandidate]`

- [ ] Write failing tests for adding inverse angles and deduplicating near-duplicates.
- [ ] Run `py -3.10 -m pytest tests/test_decision.py`.
- [ ] Implement candidate generation.
- [ ] Run `py -3.10 -m pytest tests/test_decision.py`.

### Task 2: OCR-Only Candidate Scoring

**Files:**
- Modify: `src/medical_doc_rotation/validation.py`
- Modify: `src/medical_doc_rotation/decision.py`
- Test: `tests/test_validation.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Produces: anchor-free `score_candidate_crops`.
- Produces: `decide_by_recognition(validation_scores, config) -> RotationDecision`

- [ ] Write failing tests proving anchor keywords are ignored and highest OCR score wins.
- [ ] Run targeted tests.
- [ ] Remove anchor scoring and implement recognition-only final selection.
- [ ] Run targeted tests.

### Task 3: Model-Specific Preprocessing

**Files:**
- Modify: `src/medical_doc_rotation/triton_client.py`
- Test: `tests/test_triton_adapters.py`

**Interfaces:**
- Produces: model-specific NCHW preprocessing for deep-image, Paddle doc orientation, docTR page orientation, and OCR recognition.

- [ ] Write failing tests for deep-image ImageNet normalization and per-model input sizes.
- [ ] Run targeted tests.
- [ ] Implement model-specific preprocessing.
- [ ] Run targeted tests.

### Task 4: Debug Trace Output

**Files:**
- Modify: `src/medical_doc_rotation/types.py`
- Modify: `src/medical_doc_rotation/pipeline.py`
- Modify: `src/medical_doc_rotation/cli.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: pipeline result trace fields.
- Produces: CLI print output with model probabilities, candidates, validation scores, and final decision.

- [ ] Write failing tests for trace data and CLI flag behavior.
- [ ] Run targeted tests.
- [ ] Add trace data and print formatting.
- [ ] Run targeted tests.

### Task 5: Verification and Benchmark Harness

**Files:**
- Create: `scripts/benchmark_public_samples.py`
- Modify: `README.md`
- Test: `tests/test_setup_models.py` or a new benchmark script test if needed.

**Interfaces:**
- Produces: a script that evaluates a local folder of public/anonymized samples with synthetic rotations and reports accuracy.

- [ ] Add script and README usage.
- [ ] Run `py -3.10 -m pytest`.
- [ ] Run `py -3.10 -m medical_doc_rotation.cli --help`.
- [ ] Commit and push.
