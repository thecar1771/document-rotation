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

## Run One Image

```bash
medical-doc-rotate input.jpg output.jpg --triton-url localhost:8000 --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt --model-io-path ./triton_model_repository/MODEL_IO.json
```

## Runtime Tuning

For a more aggressive operating point when rotated documents are being passed through unchanged:

```bash
medical-doc-rotate input.jpg output.jpg \
  --triton-url localhost:8000 \
  --dict-path ./triton_model_repository/ocr_korean_rec/dict.txt \
  --model-io-path ./triton_model_repository/MODEL_IO.json \
  --min-ensemble-score 0.75 \
  --min-score-margin 0.10 \
  --validation-min-score 0.40 \
  --validation-min-margin 0.08
```

For stronger protection when upright documents are being rotated incorrectly, raise `--min-ensemble-score` and `--min-score-margin`, or lower `--strong-zero-score` so a confident zero-degree model vote blocks rotation sooner.

Use `--ocr-max-width` to trade validation accuracy and latency. Lower values such as `384` or `512` are faster; higher values preserve more long text in OCR crops.

## Safety Policy

The default decision is no rotation. A non-zero rotation is applied only when coarse orientation models, margin thresholds, agreement rules, and OCR crop validation support it.
