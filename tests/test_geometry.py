from PIL import Image, ImageDraw

from medical_doc_rotation.geometry import estimate_fine_angle, select_evidence_crops


def make_line_image(angle: float) -> Image.Image:
    canvas = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(canvas)
    for y in range(80, 230, 40):
        draw.line((60, y, 340, y), fill="black", width=3)
    return canvas.rotate(angle, expand=False, fillcolor="white")


def test_estimate_fine_angle_returns_correction_for_slanted_lines():
    image = make_line_image(7.0)

    estimate = estimate_fine_angle(image)

    assert estimate.confidence > 0.2
    assert 2.0 < estimate.angle < 12.0


def test_select_evidence_crops_is_deterministic_and_bounded():
    image = Image.new("RGB", (500, 300), "white")

    first = select_evidence_crops(image, count=6)
    second = select_evidence_crops(image, count=6)

    assert first == second
    assert len(first) == 6
    for left, top, right, bottom in first:
        assert 0 <= left < right <= 500
        assert 0 <= top < bottom <= 300
