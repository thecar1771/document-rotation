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
