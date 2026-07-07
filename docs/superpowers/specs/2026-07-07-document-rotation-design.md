# Medical Document Image Rotation Design

Date: 2026-07-07
Status: Draft for user review

## Summary

Build a Python pre-processing module that runs before the existing medical document classification/extraction pipeline. The module estimates the safest rotation angle for an uploaded image, rotates the original image, saves the rotated image, and then lets the existing process continue with the saved file.

The primary objective is to improve downstream document classification/extraction. OCR improvement is secondary. The most important safety rule is to avoid rotating a document that is already correct or ambiguous.

## Inputs

The input is an image file of a medical document. Sources include phone photos, camera photos, faxes, and scans. Document types include treatment receipts, payment confirmations, detail statements, pharmacy receipts, operation records, and other table-heavy medical documents.

The images may contain:

- Heavy background noise from phone photos
- Uneven brightness and shadows
- Low-resolution or broken pixels from fax/scan input
- Structured, semi-structured, and unstructured tables
- Arbitrary rotation such as 45 degrees or 195 degrees
- Standard coarse rotations: 0, 90, 180, 270 degrees

## Goals

- Add the module only at the front of the existing process.
- Preserve the existing downstream API/process contract: downstream receives only a saved image path.
- Finish within a target maximum processing time of under 1 second per image.
- Use GPU and Triton Inference Server for model serving.
- Download and configure all required models in the Triton model repository.
- Prefer Apache-2.0 licensed models and components whenever possible.
- Use conservative decision rules so wrong rotations are rare.
- Support arbitrary angle correction by estimating fine angle and rotating the original image.
- Use extraction-oriented validation to reduce 90 vs 270 mistakes.

## Non-Goals

- Do not change the existing classification/extraction process.
- Do not pass metadata such as confidence, selected angle, or reason to downstream.
- Do not crop the final output to the document region.
- Do not perform perspective correction or document unwarping in this phase.
- Do not train a new model in the initial implementation.

## Architecture

```text
input image
-> ImageLoader
-> Preprocessor
-> FineAngleEstimator
-> OrientationEnsemble
-> AngleCandidateSelector
-> CropValidationScorer
-> FinalAngleDecider
-> ImageRotator
-> saved rotated image
-> existing downstream process
```

The module uses a low-resolution working image for inference and validation. The final rotation is applied to the original image and saved as the image that downstream consumes.

## Components

### ImageLoader

- Loads the input image.
- Applies EXIF orientation normalization if metadata is present.
- Keeps the original image available for final rotation.

### Preprocessor

- Creates a working image with a bounded long edge, initially 1024 to 1600 pixels.
- Applies lightweight normalization for contrast, brightness, and noise.
- Does not modify the original image.

### FineAngleEstimator

- Uses OpenCV-based geometry signals to estimate arbitrary skew/rotation angle.
- Uses table lines, text line orientation, connected components, projection profiles, and line clustering.
- Produces `fine_angle` and `fine_confidence`.
- If confidence is low, fine angle is ignored.

### OrientationEnsemble

Uses three coarse-orientation models to estimate one of `0, 90, 180, 270`.
The ensemble output is normalized to a correction angle, meaning the angle that should be applied to the image to make the document upright.

Initial model set:

- `deep-image-orientation-detection`
- `Paddle doc_ori`
- `docTR page orientation`

Initial model weights:

- `docTR page orientation`: 0.40
- `deep-image-orientation-detection`: 0.35
- `Paddle doc_ori`: 0.25

Paddle gets a lower weight because prior tests showed weaker performance in this domain.

### AngleCandidateSelector

- Combines coarse orientation with fine angle.
- Produces Top-K candidate angles for validation.
- Usually returns 1 or 2 candidates.
- Returns up to 3 candidates in ambiguous cases.
- Always includes both `90` and `270` when they conflict.
- Includes `0` when the no-rotation hypothesis is strong.

### CropValidationScorer

Validates candidate angles with deterministic crops and fast OCR recognition.

The scorer does not run full-page detection plus recognition for every candidate. Instead:

- Rotate the working image by each candidate angle.
- Generate deterministic evidence crops, not random crops.
- Prefer crops from text-dense, table-dense, central, upper, lower, and right-side regions.
- Use PaddleOCR ONNX Korean recognition-only inference in batch.
- Score each candidate by OCR confidence and document-like output quality.

Initial crop policy:

- Candidate angles: max 2 normally, max 3 when ambiguous.
- Crops per candidate: 8 to 12.
- Crop generation is deterministic for repeatable decisions.

Validation score:

```text
validation_score =
  avg_ocr_confidence
+ recognized_korean_numeric_ratio
+ money_date_pattern_hits
+ table_line_alignment_score
- broken_token_penalty
```

Anchor and pattern examples:

- Korean medical/payment tokens: 진료비, 본인부담, 납입, 영수증, 세부내역, 합계
- Numeric patterns: money amounts, dates, business registration numbers, phone numbers

### FinalAngleDecider

Default decision is always `0`, meaning no rotation.

A non-zero rotation is applied only when:

- Ensemble best angle is not 0.
- Ensemble best score is at least 0.85.
- Best score minus second score is at least 0.20.
- At least two coarse models agree on the same direction.
- Validation supports the candidate or does not contradict it.
- The `0` candidate is not strongly supported.

Special cases:

- If `90` and `270` conflict, both are validated.
- If validation cannot clearly distinguish candidates, keep original orientation.
- Apply `180` only when validation strongly supports upside-down correction.
- Use fine angle only when OpenCV confidence is high.

Final angle:

```text
final_angle = coarse_angle + fine_angle
```

`coarse_angle` and `fine_angle` are both correction angles to apply to the original image, not merely observed document angles. The implementation must normalize model-specific angle conventions at the adapter boundary.

Example:

```text
coarse_angle = 270
fine_angle = -3.2
final_angle = 266.8
```

### ImageRotator

- Applies the final angle to the original image.
- Saves the rotated image.
- Does not crop to the document region.
- Allows output width/height to change naturally after rotation.
- Avoids cutting off image content when rotating by arbitrary angles.

## Triton Serving Design

The Triton server already exists. This work must prepare the model repository layout, model configs, and download scripts so the existing Triton server can load the required models.

All ML model inference used by the online request path must be served through Triton. Local Python/OpenCV code is allowed only for deterministic image processing, geometry analysis, candidate generation, final decision rules, and final image rotation.

Required served models:

```text
triton_model_repository/
  orientation_deep_image/
    1/model.onnx
    config.pbtxt
  orientation_paddle_doc_ori/
    1/model.onnx
    config.pbtxt
  orientation_doctr_page/
    1/model.onnx
    config.pbtxt
  ocr_korean_rec/
    1/model.onnx
    config.pbtxt
```

Optional Python backend models may be added for preprocessing/postprocessing if ONNX-only configuration becomes too rigid:

```text
triton_model_repository/
  rotation_preprocess/
  orientation_fusion/
  ocr_validation_postprocess/
```

Preferred serving pattern:

- Use ONNX Runtime backend for ONNX models.
- Use Python code in the application for OpenCV fine-angle estimation and final decision logic.
- Keep Triton ensemble usage optional for the initial version because the application needs dynamic Top-K candidate generation and deterministic crop creation.
- Add a Triton ensemble later if stable request shapes make it worthwhile.

## Model Download Requirements

The implementation must include an explicit model download/setup command or script. It should:

- Download each required model into a local model cache.
- Export or convert models when the source project does not provide the exact ONNX artifact required by Triton.
- Copy or convert models into the Triton repository layout.
- Generate or install `config.pbtxt` files.
- Validate that each model is available from Triton before the application runs.
- Avoid downloading models during request handling.

Initial sources to use or verify:

- PaddleOCR / Paddle ONNX models: Apache-2.0 preferred.
- docTR: Apache-2.0. If a ready ONNX artifact is not available for the selected page-orientation model, the setup script must export the model and record the source version.
- PaddleOCR ONNX Korean recognition: Apache-2.0 according to the model card, but must be verified during setup.
- `deep-image-orientation-detection`: Hugging Face currently marks this model as MIT. This is permissive but not Apache-2.0. Since the model is already available in the current server environment, treat it as an accepted permissive-license exception unless the team requires strict Apache-2.0 only. If strict Apache-2.0 is required, benchmark `ternaus/check_orientation` as the replacement candidate.

## Runtime Flow

```text
1. Load image and normalize EXIF orientation.
2. Create bounded working image.
3. Run OpenCV fine-angle estimation locally.
4. Call Triton coarse orientation models.
5. Fuse coarse model probabilities.
6. Select Top-K candidate angles.
7. Generate deterministic crops for each candidate.
8. Call Triton OCR Korean recognition model in batch.
9. Score validation outputs.
10. Decide final angle with conservative no-op rules.
11. Rotate original image and save result.
12. Call existing downstream process with the saved image.
```

## Performance Strategy

- No model cold-start during request handling.
- Preload Triton models before traffic.
- Use resized working images for inference and validation.
- Use recognition-only validation, not full detection plus recognition.
- Batch OCR crop recognition calls per request.
- Limit candidate count and crop count.
- Track p50, p95, and p99 latency.

Initial limits:

```text
max_working_long_edge: 1600
normal_candidate_count: 2
max_candidate_count: 3
crop_count_per_candidate: 8 to 12
ensemble_timeout_ms: 250
validation_timeout_ms: 350
total_target_ms: 1000
```

## Evaluation Dataset

Use secured operating samples, divided into:

- Correctly oriented documents
- 90/180/270 rotated documents
- 90 vs 270 confusion cases
- Arbitrary angle cases such as 45 or 195 degrees
- Phone photos with heavy backgrounds
- Low-quality fax/scan images
- Table-heavy medical fee documents
- Ambiguous images that should remain unchanged

## Evaluation Metrics

Primary:

- Wrong-rotation rate for already-correct documents
- Downstream classification/extraction improvement
- 90 vs 270 confusion rate
- p95 latency under 1 second

Secondary:

- Rotation success rate for rotated documents
- Fallback/no-op rate
- Validation override accuracy
- p99 latency
- Model disagreement rate

Initial target:

```text
wrong_rotation_rate: <= 0.1%
p95_latency: < 1000 ms
90_270_confusion: tracked separately and reduced with validation
```

## Testing

Unit tests:

- EXIF normalization
- working image resize policy
- fine-angle estimator confidence handling
- ensemble probability fusion
- Top-K candidate selection
- no-op protection rules
- deterministic crop generation
- OCR validation scoring
- final angle decision
- arbitrary-angle rotation output dimensions

Integration tests:

- Triton model readiness checks
- inference calls for all served models
- full request flow on sample images
- model timeout and fallback behavior
- saved output file correctness

Benchmark tests:

- p50/p95/p99 latency on operating samples
- candidate count and crop count sensitivity
- GPU utilization under concurrency
- cold-start exclusion verification

## Operational Notes

Even though downstream cannot accept metadata, the preprocessor should log internal decision data for monitoring:

- input image size
- selected final angle
- coarse model scores
- fine angle and confidence
- validation scores
- whether validation overrode ensemble
- latency by stage
- fallback reason

Logs must not include raw medical image content or OCR text unless explicitly approved by the data governance policy.

## Future Work

- Perspective correction / document rectification.
- Self-trained medical document orientation model.
- Distillation from the ensemble into one smaller model.
- Automatic threshold tuning from operating feedback.
- Triton ensemble model after request shapes stabilize.
