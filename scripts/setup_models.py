import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    repo_id: str
    filename: str
    license_name: str


@dataclass(frozen=True)
class OnnxIo:
    input_name: str
    input_shape: list[int]
    output_name: str
    output_shape: list[int]


ARTIFACTS = [
    ModelArtifact(
        "orientation_deep_image",
        "DuarteBarbosa/deep-image-orientation-detection",
        "orientation_model_v2_0.9882.onnx",
        "MIT",
    ),
    ModelArtifact(
        "orientation_doctr_page",
        "Felix92/onnxtr-mobilenet-v3-small-page-orientation",
        "model.onnx",
        "Apache-2.0",
    ),
    ModelArtifact(
        "orientation_paddle_doc_ori",
        "monkt/paddleocr-onnx",
        "preprocessing/doc-orientation/model.onnx",
        "Apache-2.0",
    ),
    ModelArtifact(
        "ocr_korean_rec",
        "monkt/paddleocr-onnx",
        "languages/korean/rec.onnx",
        "Apache-2.0",
    ),
]


def resolve_repo_file(repo_id: str, requested: str) -> str:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id)
    return resolve_file_from_listing(files, requested)


def resolve_file_from_listing(files: Iterable[str], requested: str) -> str:
    file_list = list(files)
    if requested in file_list:
        return requested
    basename = Path(requested).name
    basename_matches = [item for item in file_list if item.endswith(basename)]
    if basename_matches:
        return basename_matches[0]
    lowered = [(item, item.lower()) for item in file_list]
    if "doc-orientation" in requested or "doc_ori" in requested:
        matches = [
            item
            for item, low in lowered
            if low.endswith(".onnx") and ("doc" in low and ("ori" in low or "orientation" in low))
        ]
        if matches:
            return matches[0]
    if "korean" in requested and requested.endswith(".onnx"):
        matches = [
            item
            for item, low in lowered
            if low.endswith(".onnx") and "korean" in low and ("rec" in low or "recognition" in low)
        ]
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve {requested}")


def resolve_dictionary_file(repo_id: str) -> str:
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id)
    candidates = [
        item
        for item in files
        if "korean" in item.lower()
        and ("dict" in item.lower() or item.lower().endswith((".txt", ".dict")))
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not resolve Korean OCR dictionary in {repo_id}")
    return candidates[0]


def download_artifact(artifact: ModelArtifact, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = resolve_repo_file(artifact.repo_id, artifact.filename)
    downloaded = hf_hub_download(repo_id=artifact.repo_id, filename=filename, cache_dir=cache_dir)
    return Path(downloaded)


def _shape_from_value_info(value_info) -> list[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


def inspect_onnx_io(model_path: Path) -> OnnxIo:
    import onnx

    model = onnx.load(model_path)
    graph_input = next(item for item in model.graph.input if item.name not in {init.name for init in model.graph.initializer})
    graph_output = model.graph.output[0]
    return OnnxIo(
        input_name=graph_input.name,
        input_shape=_shape_from_value_info(graph_input),
        output_name=graph_output.name,
        output_shape=_shape_from_value_info(graph_output),
    )


def _triton_dims(shape: list[int]) -> list[int]:
    if len(shape) >= 2 and shape[0] in {-1, 0, 1}:
        return shape[1:]
    return shape


def write_model_config(
    model_dir: Path,
    name: str,
    input_name: str,
    input_shape: list[int],
    output_name: str,
    output_dims: list[int],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    input_dims = ", ".join(str(item) for item in _triton_dims(input_shape))
    output_shape = ", ".join(str(item) for item in _triton_dims(output_dims))
    text = f'''name: "{name}"
backend: "onnxruntime"
max_batch_size: 8
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {input_dims} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {output_shape} ]
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
    from huggingface_hub import hf_hub_download

    repo_dir.mkdir(parents=True, exist_ok=True)
    io_manifest: dict[str, dict[str, str]] = {}
    for artifact in ARTIFACTS:
        source = download_artifact(artifact, cache_dir)
        version_dir = repo_dir / artifact.name / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        model_path = version_dir / "model.onnx"
        shutil.copy2(source, model_path)
        io = inspect_onnx_io(model_path)
        write_model_config(
            repo_dir / artifact.name,
            name=artifact.name,
            input_name=io.input_name,
            input_shape=io.input_shape,
            output_name=io.output_name,
            output_dims=io.output_shape,
        )
        io_manifest[artifact.name] = {"input": io.input_name, "output": io.output_name}
        if artifact.name == "ocr_korean_rec":
            dict_file = resolve_dictionary_file(artifact.repo_id)
            dict_source = Path(hf_hub_download(artifact.repo_id, dict_file, cache_dir=cache_dir))
            shutil.copy2(dict_source, repo_dir / artifact.name / "dict.txt")
    write_repository_manifest(repo_dir, ARTIFACTS)
    (repo_dir / "MODEL_IO.json").write_text(json.dumps(io_manifest, indent=2) + "\n", encoding="utf-8")


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
