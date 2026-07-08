from pathlib import Path

import onnx
from onnx import TensorProto, helper

from scripts.setup_models import (
    ModelArtifact,
    configure_existing_repository,
    find_existing_model_source,
    write_model_config,
    write_repository_manifest,
)


def write_tiny_onnx(path: Path, input_name: str = "input", output_name: str = "output") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", [input_name], [output_name])],
        name="unit",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 16, 16])],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)


def test_write_model_config_creates_onnxruntime_config(tmp_path: Path):
    model_dir = tmp_path / "orientation_deep_image"

    write_model_config(
        model_dir,
        name="orientation_deep_image",
        input_name="input",
        input_shape=[1, 3, 512, 512],
        output_name="probabilities",
        output_dims=[-1],
    )

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


def test_find_existing_model_source_prefers_triton_version_path(tmp_path: Path):
    artifact = ModelArtifact("unit_model", "repo/name", "nested/model.onnx", "Apache-2.0")
    expected = tmp_path / "unit_model" / "1" / "model.onnx"
    write_tiny_onnx(expected)
    fallback = tmp_path / "nested" / "model.onnx"
    write_tiny_onnx(fallback)

    assert find_existing_model_source(tmp_path, artifact) == expected


def test_configure_existing_repository_rewrites_configs_without_downloading(tmp_path: Path):
    artifacts = [
        ModelArtifact("orientation_deep_image", "repo/deep", "deep.onnx", "MIT"),
        ModelArtifact("ocr_korean_rec", "repo/ocr", "languages/korean/rec.onnx", "Apache-2.0"),
    ]
    write_tiny_onnx(tmp_path / "orientation_deep_image" / "1" / "model.onnx", "deep_input", "deep_output")
    write_tiny_onnx(tmp_path / "languages" / "korean" / "rec.onnx", "ocr_input", "ocr_output")
    (tmp_path / "languages" / "korean" / "dict.txt").write_text("ga\nna\n", encoding="utf-8")

    configure_existing_repository(tmp_path, artifacts)

    assert (tmp_path / "orientation_deep_image" / "config.pbtxt").exists()
    assert (tmp_path / "ocr_korean_rec" / "config.pbtxt").exists()
    assert (tmp_path / "ocr_korean_rec" / "1" / "model.onnx").exists()
    assert (tmp_path / "ocr_korean_rec" / "dict.txt").read_text(encoding="utf-8") == "ga\nna\n"

    manifest = (tmp_path / "MODEL_IO.json").read_text(encoding="utf-8")
    assert "deep_input" in manifest
    assert "ocr_output" in manifest
