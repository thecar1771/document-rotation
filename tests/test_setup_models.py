from pathlib import Path

from scripts.setup_models import ModelArtifact, write_model_config, write_repository_manifest


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
