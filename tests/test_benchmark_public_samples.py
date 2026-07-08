from pathlib import Path

from PIL import Image

from scripts.benchmark_public_samples import (
    angle_passes,
    expected_correction,
    iter_image_paths,
    parse_rotations,
)


def test_parse_rotations_accepts_comma_separated_degrees():
    assert parse_rotations("0,90, 195") == [0.0, 90.0, 195.0]


def test_expected_correction_is_inverse_of_applied_rotation():
    assert expected_correction(90.0) == 270.0
    assert expected_correction(270.0) == 90.0
    assert expected_correction(0.0) == 0.0


def test_angle_passes_handles_wraparound():
    assert angle_passes(359.0, 1.0, tolerance_degrees=3.0)
    assert not angle_passes(350.0, 10.0, tolerance_degrees=5.0)


def test_iter_image_paths_filters_supported_images(tmp_path: Path):
    Image.new("RGB", (10, 10), "white").save(tmp_path / "a.png")
    Image.new("RGB", (10, 10), "white").save(tmp_path / "b.jpg")
    (tmp_path / "c.txt").write_text("nope", encoding="utf-8")

    assert [path.name for path in iter_image_paths(tmp_path)] == ["a.png", "b.jpg"]
