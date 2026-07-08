import math

import cv2
import numpy as np
from PIL import Image

from medical_doc_rotation.image_ops import pil_to_rgb_array
from medical_doc_rotation.types import FineAngleEstimate


def _normalize_line_angle(angle: float) -> float:
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def estimate_fine_angle(image: Image.Image) -> FineAngleEstimate:
    rgb = pil_to_rgb_array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(30, image.width // 8),
        maxLineGap=12,
    )
    if lines is None:
        return FineAngleEstimate(angle=0.0, confidence=0.0)

    weighted_angles: list[tuple[float, float]] = []
    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue
        raw_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        normalized = _normalize_line_angle(raw_angle)
        if abs(normalized) <= 30:
            weighted_angles.append((normalized, length))

    if not weighted_angles:
        return FineAngleEstimate(angle=0.0, confidence=0.0)

    total_weight = sum(weight for _, weight in weighted_angles)
    mean_angle = sum(angle * weight for angle, weight in weighted_angles) / total_weight
    variance = sum(weight * (angle - mean_angle) ** 2 for angle, weight in weighted_angles) / total_weight
    confidence = min(1.0, len(weighted_angles) / 24.0) * max(
        0.0,
        1.0 - min(1.0, math.sqrt(variance) / 15.0),
    )
    return FineAngleEstimate(angle=-mean_angle, confidence=confidence)


def select_evidence_crops(image: Image.Image, count: int) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    crop_w = max(32, round(width * 0.42))
    crop_h = max(32, round(height * 0.18))
    anchors = [
        (0.29, 0.18),
        (0.71, 0.18),
        (0.29, 0.38),
        (0.71, 0.38),
        (0.29, 0.58),
        (0.71, 0.58),
        (0.29, 0.78),
        (0.71, 0.78),
        (0.50, 0.50),
        (0.75, 0.82),
        (0.50, 0.18),
        (0.50, 0.82),
    ]
    boxes: list[tuple[int, int, int, int]] = []
    for cx_ratio, cy_ratio in anchors[:count]:
        cx = round(width * cx_ratio)
        cy = round(height * cy_ratio)
        left = min(max(0, cx - crop_w // 2), max(0, width - crop_w))
        top = min(max(0, cy - crop_h // 2), max(0, height - crop_h))
        boxes.append((left, top, min(width, left + crop_w), min(height, top + crop_h)))
    return boxes
