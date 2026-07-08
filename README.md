# Medical Document Rotation

Python preprocessor for medical document image rotation. Orientation models propose candidate angles, Korean OCR recognition scores each candidate, the best recognized candidate is applied to the original image, and the saved image is passed to an existing downstream process.

## Install

```bash
pip install -e ".[test]"
```

## Prepare Triton Models

```bash
medical-doc-setup-models --repo-dir ./triton_model_repository --cache-dir ./.model-cache
```

Point the existing Triton server at `./triton_model_repository` or copy the generated model directories into the server's configured model repository.

Generated repository layout:

```text
triton_model_repository/
  orientation_deep_image/
  orientation_paddle_doc_ori/
  orientation_doctr_page/
  ocr_korean_rec/
  MODEL_IO.json
  MODEL_SOURCES.md
```

`MODEL_IO.json` records the exact ONNX input and output tensor names discovered during setup.

## Regenerate Triton Configs Without Downloading

If the ONNX files are already present, regenerate only Triton server configuration:

```bash
medical-doc-setup-models --repo-dir ./triton_model_repository --config-only
```

If the raw downloaded model files are in a separate folder, copy them into the Triton layout and generate configs without network access:

```bash
medical-doc-setup-models \
  --repo-dir ./triton_model_repository \
  --source-dir ./downloaded-models \
  --config-only
```

This rewrites each model's `config.pbtxt`, `MODEL_IO.json`, and `MODEL_SOURCES.md` from local ONNX metadata. For Korean OCR, it also copies a local `dict.txt` when found.

## Write Default Triton Config Templates

To create only the known per-model `config.pbtxt` structure without downloading models or inspecting ONNX files:

```bash
medical-doc-setup-models --repo-dir ./triton_model_repository --write-default-configs
```

This creates:

```text
triton_model_repository/
  MODEL_IO.json
  MODEL_SOURCES.md
  orientation_deep_image/config.pbtxt
  orientation_doctr_page/config.pbtxt
  orientation_paddle_doc_ori/config.pbtxt
  ocr_korean_rec/config.pbtxt
```

Then place each ONNX file at `<model_name>/1/model.onnx` before starting Triton.

## Run One Image

```bash
medical-doc-rotate input.jpg output.jpg --triton-url localhost:8000 --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt --model-io-path ./triton_model_repository/MODEL_IO.json
```

The command prints a trace for debugging:

```text
model=deep_image scores=90.00:0.9123,270.00:0.0412,...
candidates=90.00,270.00,0.00,180.00
angle=90.00 score=2.143 avg_conf=0.921 recognized_ratio=0.742 recognized_chars=84 broken_penalty=0.000
final angle=90.00 rotate=True reason=recognition_best
```

## Runtime Tuning

Candidate generation is intentionally simple:

- take the top orientation angles from each model
- add the inverse angle for each model angle
- add fine-angle and inverse fine-angle when OpenCV confidence is high
- deduplicate nearby angles
- pick the candidate with the highest OCR recognition score

To widen the search when model direction is unstable:

```bash
medical-doc-rotate input.jpg output.jpg \
  --triton-url localhost:8000 \
  --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt \
  --model-io-path ./triton_model_repository/MODEL_IO.json \
  --candidate-top-k 3 \
  --candidate-dedupe-degrees 3 \
  --crops-per-candidate 12 \
  --ocr-max-width 512
```

Use `--ocr-max-width` to trade validation accuracy and latency. Lower values such as `384` or `512` are faster; higher values preserve more long text in OCR crops.

## Benchmark Public Samples

Place public or anonymized upright samples in a folder. The benchmark creates synthetic rotations, runs the full Triton-backed pipeline, and checks whether the predicted correction matches the known inverse angle.

```bash
medical-doc-benchmark-public-samples \
  --samples-dir ./public-medical-samples \
  --work-dir ./benchmark-work \
  --triton-url localhost:8000 \
  --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt \
  --model-io-path ./triton_model_repository/MODEL_IO.json \
  --limit 50 \
  --rotations 0,90,180,270 \
  --json-output ./benchmark-work/results.json
```

The script exits with code `0` only when every generated case passes.

## Safety Policy

This repository does not scrape random real medical documents. Use public, synthetic, or anonymized samples for benchmarks.
