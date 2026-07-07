from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def save_image(image: Image.Image, path: Path | str, quality: int = 95) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=quality, subsampling=0)
    else:
        image.save(output_path)
    return output_path


def resize_for_working(image: Image.Image, max_long_edge: int) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image.copy()
    scale = max_long_edge / long_edge
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def rotate_image(
    image: Image.Image,
    correction_angle: float,
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    normalized = correction_angle % 360
    if normalized == 0:
        return image.copy()
    if normalized == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if normalized == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if normalized == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image.rotate(
        correction_angle,
        expand=True,
        fillcolor=fill_color,
        resample=Image.Resampling.BICUBIC,
    )


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))
