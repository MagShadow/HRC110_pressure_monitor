from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Reading:
    value: str
    confidence: float


def _find_display_region(
    hsv: np.ndarray, color: str, min_area_ratio: float = 0.01
) -> Tuple[int, int, int, int]:
    if color == "red":
        mask1 = cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 120), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "blue":
        mask = cv2.inRange(hsv, (95, 80, 80), (135, 255, 255))
    else:
        raise ValueError(f"Unknown color: {color}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No {color} region found")

    height, width = hsv.shape[:2]
    min_area = height * width * min_area_ratio
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        raise ValueError(f"No {color} region above min area")

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h


SEGMENT_MAP = {
    (1, 1, 1, 1, 1, 1, 0): "0",
    (0, 1, 1, 0, 0, 0, 0): "1",
    (1, 1, 0, 1, 1, 0, 1): "2",
    (1, 1, 1, 1, 0, 0, 1): "3",
    (0, 1, 1, 0, 0, 1, 1): "4",
    (1, 0, 1, 1, 0, 1, 1): "5",
    (1, 0, 1, 1, 1, 1, 1): "6",
    (1, 1, 1, 0, 0, 0, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}


def _segment_digit(roi: np.ndarray) -> str:
    height, width = roi.shape[:2]
    if height == 0 or width == 0:
        return "?"

    segments = []
    h = height
    w = width
    seg_w = int(w * 0.18)
    seg_h = int(h * 0.15)
    seg_center = int(h * 0.05)
    regions = [
        (seg_w, 0, w - seg_w, seg_h),  # top
        (w - seg_w, seg_h, w, h // 2 - seg_center),  # top-right
        (w - seg_w, h // 2 + seg_center, w, h - seg_h),  # bottom-right
        (seg_w, h - seg_h, w - seg_w, h),  # bottom
        (0, h // 2 + seg_center, seg_w, h - seg_h),  # bottom-left
        (0, seg_h, seg_w, h // 2 - seg_center),  # top-left
        (seg_w, h // 2 - seg_center, w - seg_w, h // 2 + seg_center),  # center
    ]

    for (x1, y1, x2, y2) in regions:
        segment = roi[y1:y2, x1:x2]
        if segment.size == 0:
            segments.append(0)
            continue
        on_ratio = cv2.countNonZero(segment) / segment.size
        segments.append(1 if on_ratio > 0.35 else 0)

    return SEGMENT_MAP.get(tuple(segments), "?")


def _extract_digits(mask: np.ndarray) -> Tuple[str, float]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > 50]
    if not boxes:
        return "", 0.0

    boxes.sort(key=lambda b: b[0])
    digits = []
    confidence = 0

    for (x, y, w, h) in boxes:
        roi = cleaned[y : y + h, x : x + w]
        if w < h * 0.25:
            if h < 0.4 * max(b[3] for b in boxes):
                digits.append(".")
            continue

        digit = _segment_digit(roi)
        digits.append(digit)
        confidence += 1 if digit != "?" else 0

    value = "".join(digits).strip(".")
    if not value:
        return "", 0.0

    return value, confidence / max(1, len(value))


def _prepare_mask(image: np.ndarray, color: str) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color == "red":
        mask1 = cv2.inRange(hsv, (0, 100, 120), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 120), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "blue":
        mask = cv2.inRange(hsv, (95, 80, 80), (135, 255, 255))
    else:
        raise ValueError(f"Unknown color: {color}")

    return mask


def read_colored_display(image: np.ndarray, color: str) -> Reading:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    x, y, w, h = _find_display_region(hsv, color=color)
    roi = image[y : y + h, x : x + w]
    mask = _prepare_mask(roi, color=color)
    value, confidence = _extract_digits(mask)
    if not value:
        raise ValueError(f"Unable to read {color} digits")
    return Reading(value=value, confidence=confidence)


def read_pressure_temperature(image_path: str) -> Tuple[Reading, Reading]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    red = read_colored_display(image, color="red")
    blue = read_colored_display(image, color="blue")
    return blue, red


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read pressure/temperature from a photo.")
    parser.add_argument("--image", required=True, help="Path to image to read.")
    args = parser.parse_args(argv)

    pressure, temperature = read_pressure_temperature(args.image)
    print(f"pressure={pressure.value} (confidence={pressure.confidence:.2f})")
    print(f"temperature={temperature.value} (confidence={temperature.confidence:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
