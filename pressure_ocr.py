from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None  # Optional fallback OCR engine.


@dataclass(frozen=True)
class Reading:
    value: str
    confidence: float


_PRESSURE_RE = re.compile(r"(\d\.?\d{2,3})")
_TEMPERATURE_RE = re.compile(r"(\d\.?\d{3,4})")


def _parse_pressure_text(text: str) -> Optional[float]:
    text = text.replace(" ", "").replace("\n", "").replace(":", ".")
    match = _PRESSURE_RE.search(text)
    if match:
        token = match.group(1)
        if "." in token:
            integer, frac = token.split(".", 1)
            if integer and len(frac) >= 2:
                return float(f"{integer[0]}.{frac[:2]}")
        digits = "".join(ch for ch in token if ch.isdigit())
        if len(digits) >= 3:
            return float(f"{digits[0]}.{digits[1:3]}")

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 3:
        return float(f"{digits[0]}.{digits[1:3]}")
    return None


def _parse_temperature_text(text: str) -> Optional[float]:
    text = text.replace(" ", "").replace("\n", "").replace(":", ".")
    match = _TEMPERATURE_RE.search(text)
    if match:
        token = match.group(1)
        if "." in token:
            integer, frac = token.split(".", 1)
            if integer and len(frac) >= 3:
                return float(f"{integer[0]}.{frac[:3]}")
        digits = "".join(ch for ch in token if ch.isdigit())
        if len(digits) >= 4:
            return float(f"{digits[0]}.{digits[1:4]}")

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return float(f"{digits[0]}.{digits[1:4]}")
    return None


def _find_pressure_screen_bbox(image: np.ndarray) -> Tuple[int, int, int, int]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, 160), (180, 100, 255))
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )

    cyan = cv2.inRange(hsv, (70, 20, 40), (125, 255, 255))
    blue = cv2.inRange(hsv, (90, 60, 70), (140, 255, 255))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Tuple[float, int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < height * width * 0.003:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 120 or h < 80:
            continue

        cyan_ratio = cv2.countNonZero(cyan[y : y + h, x : x + w]) / (w * h)
        blue_roi = blue[y : y + h, x : x + w]
        border = max(4, int(min(w, h) * 0.02))
        ring = np.zeros((h, w), dtype=np.uint8)
        ring[:border, :] = 1
        ring[-border:, :] = 1
        ring[:, :border] = 1
        ring[:, -border:] = 1
        edge_ratio = cv2.countNonZero((blue_roi > 0).astype(np.uint8) * ring) / max(
            1, int(ring.sum())
        )
        lower_bias = (y + h / 2) / height
        score = area * (1.0 + 8.0 * cyan_ratio + 4.0 * edge_ratio) * (0.3 + lower_bias)
        candidates.append((score, x, y, w, h))

    if not candidates:
        raise ValueError("Unable to locate pressure screen")
    candidates.sort(reverse=True)
    _, x, y, w, h = candidates[0]
    return x, y, w, h


def _ocr_single_digit_tesseract(mask: np.ndarray) -> Optional[str]:
    if pytesseract is None:
        return None
    canvas = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
    canvas = cv2.resize(canvas, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    votes: Counter[str] = Counter()
    for psm in (10, 13, 8):
        config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789"
        try:
            text = pytesseract.image_to_string(canvas, config=config)
        except Exception:
            continue
        for char in text:
            if char.isdigit():
                votes[char] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


def _tesseract_values(
    image: np.ndarray, parser: callable, psm_values: Iterable[int]
) -> list[float]:
    if pytesseract is None:
        return []

    values: list[float] = []
    for psm in psm_values:
        config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789."
        try:
            text = pytesseract.image_to_string(image, config=config)
        except Exception:
            continue
        value = parser(text)
        if value is not None:
            values.append(value)
    return values


def _select_mode_value(values: list[float], decimals: int) -> Tuple[float, float]:
    if not values:
        raise ValueError("No OCR candidates")
    rounded = [round(v, decimals) for v in values]
    counts = Counter(rounded)
    most_common = counts.most_common()
    if not most_common:
        raise ValueError("No OCR candidates")

    top_count = most_common[0][1]
    top_values = [value for value, count in most_common if count == top_count]
    median = float(np.median(rounded))
    best = min(top_values, key=lambda value: abs(value - median))
    confidence = top_count / max(1, len(rounded))
    return best, confidence


def _select_temperature_value(values: list[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("No OCR candidates")

    rounded = [round(v, 3) for v in values]
    counts = Counter(rounded)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], abs(item[0] - 4.0)))
    best, best_count = ranked[0]

    # Common OCR failure is 8->6 (delta ~= 0.2). If a nearby higher value has
    # near-equal support, prefer the higher value.
    for value, count in counts.items():
        if value <= best:
            continue
        if abs((value - best) - 0.2) <= 0.012 and count >= max(2, best_count - 1):
            best = value
            best_count = count

    confidence = best_count / max(1, len(rounded))
    return best, confidence


def _read_pressure_blackhat_fallback(image: np.ndarray) -> Reading:
    if pytesseract is None:
        raise ValueError("pytesseract not available")

    x, y, w, h = _find_pressure_screen_bbox(image)
    screen = image[y : y + h, x : x + w]

    frac_roi = screen[int(0.35 * h) : int(0.96 * h), int(0.00 * w) : int(0.50 * w)]
    if frac_roi.size == 0:
        raise ValueError("Fallback pressure ROI is empty")

    frac_gray = cv2.cvtColor(frac_roi, cv2.COLOR_BGR2GRAY)
    frac_blackhat = cv2.morphologyEx(
        frac_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
    )
    _, frac_binary = cv2.threshold(frac_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    frac_weights: Counter[int] = Counter()
    for prepared in (frac_binary, frac_blackhat):
        upscaled = cv2.resize(prepared, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        for psm in (6, 7, 10, 11, 13):
            config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789."
            try:
                text = pytesseract.image_to_string(upscaled, config=config).strip()
            except Exception:
                continue
            if not text:
                continue

            for match in re.finditer(r"\.(\d{2})", text):
                frac_weights[int(match.group(1))] += 3

            digits = "".join(char for char in text if char.isdigit())
            if len(digits) == 2:
                frac_weights[int(digits)] += 2
            for index in range(len(digits) - 1):
                frac_weights[int(digits[index : index + 2])] += 1

    frac_candidates = {frac: weight for frac, weight in frac_weights.items() if 0 <= frac <= 99}
    if not frac_candidates:
        raise ValueError("Fallback OCR missing pressure fraction")
    frac, frac_votes = max(frac_candidates.items(), key=lambda item: item[1])

    lead_roi = screen[int(0.28 * h) : int(0.80 * h), int(0.00 * w) : int(0.24 * w)]
    if lead_roi.size == 0:
        raise ValueError("Fallback lead-digit ROI is empty")

    lead_gray = cv2.cvtColor(lead_roi, cv2.COLOR_BGR2GRAY)
    lead_blackhat = cv2.morphologyEx(
        lead_gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
    )
    _, lead_binary = cv2.threshold(lead_blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lead_counts: Counter[int] = Counter()
    for prepared in (lead_binary, lead_blackhat):
        upscaled = cv2.resize(prepared, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
        for psm in (6, 7, 8, 10, 13):
            config = f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789"
            try:
                text = pytesseract.image_to_string(upscaled, config=config).strip()
            except Exception:
                continue
            for char in text:
                if char.isdigit():
                    digit = int(char)
                    if 0 <= digit <= 5:
                        lead_counts[digit] += 1

    if not lead_counts:
        raise ValueError("Fallback OCR missing pressure leading digit")
    lead, lead_votes = lead_counts.most_common(1)[0]

    value = lead + frac / 100.0
    if not (0.0 <= value <= 6.0):
        raise ValueError(f"Fallback pressure out-of-range value={value:.2f}")

    confidence = min(0.70, 0.25 + 0.05 * frac_votes + 0.07 * lead_votes)
    return Reading(value=f"{value:.2f}", confidence=confidence)


def _read_pressure_tesseract(image: np.ndarray) -> Reading:
    if pytesseract is None:
        raise ValueError("pytesseract not available")

    x, y, w, h = _find_pressure_screen_bbox(image)
    crop_specs = [
        (0.03, 0.60, 0.26, 0.78),
        (0.03, 0.56, 0.22, 0.90),
        (0.05, 0.60, 0.22, 0.82),
        (0.07, 0.62, 0.24, 0.82),
        (0.09, 0.58, 0.24, 0.90),
        (0.03, 0.66, 0.24, 0.78),
    ]

    values: list[float] = []
    for x1r, x2r, y1r, y2r in crop_specs:
        x1 = x + int(x1r * w)
        x2 = x + int(x2r * w)
        y1 = y + int(y1r * h)
        y2 = y + int(y2r * h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crop = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cyan_mask = cv2.inRange(hsv, (65, 15, 20), (130, 255, 255))

        for prepared in (otsu, adaptive, cyan_mask):
            prepared = cv2.resize(
                prepared, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR
            )
            values.extend(_tesseract_values(prepared, _parse_pressure_text, [6, 7, 8, 10, 13]))

    in_range = [value for value in values if 0.0 <= value <= 6.0]
    if not in_range:
        return _read_pressure_blackhat_fallback(image)

    best, confidence = _select_mode_value(in_range, decimals=2)
    if best > 2.0 and confidence < 0.75:
        try:
            return _read_pressure_blackhat_fallback(image)
        except Exception as exc:
            raise ValueError(f"Pressure OCR candidates are ambiguous ({exc})") from exc
    return Reading(value=f"{best:.2f}", confidence=confidence)


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


def _segment_digit(roi: np.ndarray, on_threshold: float = 0.35) -> str:
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
        segments.append(1 if on_ratio > on_threshold else 0)

    return SEGMENT_MAP.get(tuple(segments), "?")


_DIGIT_TEMPLATE_CACHE: Optional[dict[str, list[np.ndarray]]] = None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_on = a > 0
    b_on = b > 0
    intersection = np.logical_and(a_on, b_on).sum()
    union = np.logical_or(a_on, b_on).sum()
    return float(intersection / union) if union else 0.0


def _digit_templates() -> dict[str, list[np.ndarray]]:
    global _DIGIT_TEMPLATE_CACHE
    if _DIGIT_TEMPLATE_CACHE is not None:
        return _DIGIT_TEMPLATE_CACHE

    width, height = 32, 56
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    templates: dict[str, list[np.ndarray]] = {}
    for digit in "0123456789":
        variants: list[np.ndarray] = []
        for font in fonts:
            for scale in (1.0, 1.2, 1.4, 1.6, 1.8):
                for thickness in (1, 2, 3):
                    canvas = np.zeros((height, width), dtype=np.uint8)
                    (text_w, text_h), _ = cv2.getTextSize(digit, font, scale, thickness)
                    x = max(0, (width - text_w) // 2)
                    y = max(text_h, (height + text_h) // 2)
                    cv2.putText(
                        canvas, digit, (x, y), font, scale, 255, thickness, cv2.LINE_AA
                    )
                    _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
                    variants.append(canvas)
        templates[digit] = variants

    _DIGIT_TEMPLATE_CACHE = templates
    return templates


def _classify_digit_template(glyph: np.ndarray) -> Tuple[str, float]:
    target = cv2.resize(glyph, (32, 56), interpolation=cv2.INTER_NEAREST)
    _, target = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY)
    best_digit = "?"
    best_score = -1.0

    for digit, variants in _digit_templates().items():
        for variant in variants:
            score = _iou(target, variant)
            if score > best_score:
                best_score = score
                best_digit = digit

    return best_digit, best_score


def _largest_contour_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found")
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def _read_pressure_blue(image: np.ndarray) -> Reading:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Find the HRC controller body from its blue frame.
    blue = cv2.inRange(hsv, (90, 60, 80), (140, 255, 255))
    blue = cv2.morphologyEx(
        blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    x, y, w, h = _largest_contour_bbox(blue)
    controller_hsv = hsv[y : y + h, x : x + w]
    controller_bgr = image[y : y + h, x : x + w]

    # Locate the bright white LCD area inside the controller.
    white = cv2.inRange(controller_hsv, (0, 0, 170), (180, 90, 255))
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    sx, sy, sw, sh = _largest_contour_bbox(white)
    screen = controller_bgr[sy : sy + sh, sx : sx + sw]

    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    margin = 8
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glyph_boxes: list[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        gx, gy, gw, gh = cv2.boundingRect(contour)
        if gw > 0.8 * binary.shape[1] or gh > 0.8 * binary.shape[0]:
            continue
        if gy < 0.25 * binary.shape[0]:
            continue
        if gh < 0.12 * binary.shape[0]:
            continue
        glyph_boxes.append((gx, gy, gw, gh))

    glyph_boxes.sort(key=lambda b: b[0])
    if len(glyph_boxes) < 3:
        raise ValueError("Unable to isolate pressure digits")

    first_three = glyph_boxes[:3]
    digits: list[str] = []
    confidence = 0.0
    for gx, gy, gw, gh in first_three:
        glyph = binary[gy : gy + gh, gx : gx + gw]
        digit, score = _classify_digit_template(glyph)
        digits.append(digit)
        confidence += score

    if any(d not in "0123456789" for d in digits):
        raise ValueError("Unable to decode pressure digits")

    return Reading(value=f"{digits[0]}.{digits[1]}{digits[2]}", confidence=confidence / 3.0)


def _cluster_red_region(core_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        (core_mask > 0).astype(np.uint8), 8
    )

    components: list[Tuple[int, int, int, int, int, float, float]] = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < 8:
            continue
        cx, cy = centroids[idx]
        components.append((x, y, w, h, area, float(cx), float(cy)))

    if not components:
        return None

    adjacency: list[set[int]] = [set() for _ in range(len(components))]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            xi, yi, wi, hi, _, cxi, cyi = components[i]
            xj, yj, wj, hj, _, cxj, cyj = components[j]
            distance = ((cxi - cxj) ** 2 + (cyi - cyj) ** 2) ** 0.5
            threshold = max(80.0, 2.2 * max(wi, hi, wj, hj))
            if distance <= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = [False] * len(components)
    clusters: list[Tuple[float, int, int, int, int, int]] = []
    for idx in range(len(components)):
        if visited[idx]:
            continue
        stack = [idx]
        visited[idx] = True
        cluster_indices: list[int] = []
        while stack:
            current = stack.pop()
            cluster_indices.append(current)
            for neighbour in adjacency[current]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)

        xs = [components[k][0] for k in cluster_indices]
        ys = [components[k][1] for k in cluster_indices]
        x2s = [components[k][0] + components[k][2] for k in cluster_indices]
        y2s = [components[k][1] + components[k][3] for k in cluster_indices]
        total_area = sum(components[k][4] for k in cluster_indices)
        bx = min(xs)
        by = min(ys)
        bw = max(x2s) - bx
        bh = max(y2s) - by
        aspect = bw / max(1, bh)
        score = total_area * (1.0 + 0.3 * aspect)
        clusters.append((score, total_area, bx, by, bw, bh))

    clusters.sort(reverse=True)
    for _, _, bx, by, bw, bh in clusters:
        if bw > 150 and bh > 70 and bw / max(1, bh) > 1.5:
            return bx, by, bw, bh

    _, _, bx, by, bw, bh = clusters[0]
    return bx, by, bw, bh


def _merge_vertical_split_boxes(boxes: list[Tuple[int, int, int, int]]) -> list[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged: list[list[int]] = []
    for x, y, w, h in boxes:
        if not merged:
            merged.append([x, y, w, h])
            continue
        px, py, pw, ph = merged[-1]
        overlap = max(0, min(x + w, px + pw) - max(x, px))
        min_width = min(w, pw)
        vertical_gap = max(0, max(y, py) - min(y + h, py + ph))
        if min_width > 0 and overlap / min_width > 0.5 and vertical_gap <= 8:
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            merged.append([x, y, w, h])
    return [tuple(item) for item in merged]


def _decode_red_digit(mask: np.ndarray) -> Tuple[str, float]:
    h, w = mask.shape[:2]
    aspect = w / max(1, h)
    if aspect < 0.30:
        return "1", 0.90

    digit = _segment_digit(mask)
    if digit != "?":
        return digit, 0.90

    return _classify_digit_template(mask)


def _decode_red_digit_recover(mask: np.ndarray) -> Tuple[str, float]:
    h, w = mask.shape[:2]
    aspect = w / max(1, h)
    if aspect < 0.30:
        return "1", 0.90

    for threshold, confidence in ((0.35, 0.90), (0.22, 0.80), (0.15, 0.70)):
        digit = _segment_digit(mask, on_threshold=threshold)
        if digit != "?":
            return digit, confidence

    digit = _ocr_single_digit_tesseract(mask)
    if digit is not None:
        return digit, 0.72

    return _classify_digit_template(mask)


def _read_temperature_red(image: np.ndarray) -> Reading:
    b, g, r = cv2.split(image)
    red_core = ((r > 220) & (r > g + 40) & (r > b + 40)).astype(np.uint8) * 255
    red_core[int(image.shape[0] * 0.35) :, :] = 0

    red_region = _cluster_red_region(red_core)
    if red_region is None:
        raise ValueError("Unable to locate red temperature display")

    x, y, w, h = red_region
    roi = image[y : y + h, x : x + w]
    rb, rg, rr = cv2.split(roi)
    base_mask = ((rr > 220) & (rr > rg + 40) & (rr > rb + 40)).astype(np.uint8) * 255

    group_mask = cv2.dilate(
        base_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1
    )
    group_mask = cv2.morphologyEx(
        group_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes: list[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        gx, gy, gw, gh = cv2.boundingRect(contour)
        if gh < 0.3 * h:
            continue
        candidate_boxes.append((gx, gy, gw, gh))

    merged_boxes = _merge_vertical_split_boxes(candidate_boxes)
    merged_boxes = [box for box in merged_boxes if box[0] < 0.65 * w]
    merged_boxes.sort(key=lambda box: box[0])
    if len(merged_boxes) < 3:
        raise ValueError("Unable to isolate temperature digits")

    selected_boxes = merged_boxes[:4] if len(merged_boxes) >= 4 else merged_boxes[:3]
    digits: list[str] = []
    confidence = 0.0
    for gx, gy, gw, gh in selected_boxes:
        glyph = base_mask[gy : gy + gh, gx : gx + gw]
        digit, score = _decode_red_digit(glyph)
        digits.append(digit)
        confidence += score

    if any(d not in "0123456789" for d in digits):
        raise ValueError("Unable to decode temperature digits")

    if len(digits) >= 4:
        value = f"{digits[0]}.{digits[1]}{digits[2]}{digits[3]}"
    else:
        value = f"{digits[0]}.{digits[1]}{digits[2]}"
    return Reading(value=value, confidence=confidence / len(digits))


def _read_temperature_red_recover(image: np.ndarray) -> Reading:
    b, g, r = cv2.split(image)
    red_core = ((r > 220) & (r > g + 40) & (r > b + 40)).astype(np.uint8) * 255
    red_core[int(image.shape[0] * 0.35) :, :] = 0

    red_region = _cluster_red_region(red_core)
    if red_region is None:
        raise ValueError("Unable to locate red temperature display")

    x, y, w, h = red_region
    roi = image[y : y + h, x : x + w]
    rb, rg, rr = cv2.split(roi)
    base_mask = ((rr > 220) & (rr > rg + 40) & (rr > rb + 40)).astype(np.uint8) * 255

    group_mask = cv2.dilate(
        base_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1
    )
    group_mask = cv2.morphologyEx(
        group_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes: list[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        gx, gy, gw, gh = cv2.boundingRect(contour)
        if gh < 0.3 * h:
            continue
        candidate_boxes.append((gx, gy, gw, gh))

    merged_boxes = _merge_vertical_split_boxes(candidate_boxes)
    merged_boxes = [box for box in merged_boxes if box[0] < 0.65 * w]
    merged_boxes.sort(key=lambda box: box[0])
    if len(merged_boxes) < 4:
        raise ValueError("Unable to isolate temperature digits in recover mode")

    first_four = merged_boxes[:4]
    digits: list[str] = []
    confidence = 0.0
    for gx, gy, gw, gh in first_four:
        glyph = base_mask[gy : gy + gh, gx : gx + gw]
        digit, score = _decode_red_digit_recover(glyph)
        digits.append(digit)
        confidence += score

    if any(d not in "0123456789" for d in digits):
        raise ValueError("Unable to decode temperature digits in recover mode")

    return Reading(value=f"{digits[0]}.{digits[1]}{digits[2]}{digits[3]}", confidence=confidence / 4.0)


def _read_temperature_tesseract(image: np.ndarray) -> Reading:
    if pytesseract is None:
        raise ValueError("pytesseract not available")

    height, width = image.shape[:2]
    top = image[: int(height * 0.55), :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 25, 70), (20, 255, 255))
    red2 = cv2.inRange(hsv, (160, 25, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Tuple[float, int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < top.shape[0] * top.shape[1] * 0.001:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 100 or h < 40:
            continue
        fill_ratio = cv2.countNonZero(red_mask[y : y + h, x : x + w]) / (w * h)
        aspect = w / h
        score = area * (1.0 + 2.0 * fill_ratio) * (0.5 + aspect)
        candidates.append((score, x, y, w, h))
    candidates.sort(reverse=True)

    values: list[float] = []
    for _, x, y, w, h in candidates[:8]:
        region = top[y : y + h, x : x + w]
        sub_specs = [
            (0.00, 0.90, 0.00, 0.75),
            (0.05, 0.90, 0.02, 0.70),
            (0.10, 0.85, 0.02, 0.70),
        ]
        for x1r, x2r, y1r, y2r in sub_specs:
            x1 = int(x1r * w)
            x2 = int(x2r * w)
            y1 = int(y1r * h)
            y2 = int(y2r * h)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = region[y1:y2, x1:x2]

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            local_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(local_hsv, (0, 35, 70), (20, 255, 255))
            m2 = cv2.inRange(local_hsv, (160, 35, 70), (180, 255, 255))
            red_local = cv2.bitwise_or(m1, m2)

            for prepared in (otsu, red_local):
                prepared = cv2.resize(
                    prepared, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR
                )
                values.extend(
                    _tesseract_values(prepared, _parse_temperature_text, [6, 7, 8, 10, 13])
                )

    in_range = [value for value in values if 2.0 <= value <= 6.0]
    if not in_range:
        raise ValueError("No valid temperature OCR candidates")

    best, confidence = _select_temperature_value(in_range)
    return Reading(value=f"{best:.3f}", confidence=confidence)


def _read_pressure(image: np.ndarray) -> Reading:
    # Try OCR engine first (if available) and fall back to legacy path.
    errors: list[str] = []
    try:
        reading = _read_pressure_tesseract(image)
        return reading
    except Exception as exc:
        errors.append(str(exc))

    try:
        return _read_pressure_blue(image)
    except Exception as exc:
        errors.append(str(exc))

    raise ValueError("Unable to decode pressure value: " + " | ".join(errors))


def _read_temperature(image: np.ndarray) -> Reading:
    errors: list[str] = []
    try:
        reading = _read_temperature_red(image)
        value = float(reading.value)
        if 2.0 <= value <= 6.0:
            return reading
        errors.append(f"legacy out-of-range temperature={reading.value}")
    except Exception as exc:
        errors.append(str(exc))

    try:
        return _read_temperature_tesseract(image)
    except Exception as exc:
        errors.append(str(exc))

    try:
        reading = _read_temperature_red_recover(image)
        value = float(reading.value)
        if 2.0 <= value <= 6.0:
            return reading
        errors.append(f"recover out-of-range temperature={reading.value}")
    except Exception as exc:
        errors.append(str(exc))

    raise ValueError("Unable to decode temperature value: " + " | ".join(errors))


def read_pressure_temperature(image_path: str) -> Tuple[Reading, Reading]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    pressure = _read_pressure(image)
    temperature = _read_temperature(image)
    return pressure, temperature


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
