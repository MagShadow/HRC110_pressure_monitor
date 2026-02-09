from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    import google.generativeai as genai
except Exception as exc:  # pragma: no cover - optional dependency
    genai = None
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None


DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_INPUT_PRICE_PER_MILLION = 0.35
DEFAULT_OUTPUT_PRICE_PER_MILLION = 1.05


@dataclass(frozen=True)
class GeminiOCRResult:
    pressure: Optional[float]
    temperature: Optional[float]
    pressure_unit: str
    temperature_unit: str
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    raw_response: str


_JSON_START_RE = re.compile(r"\{", re.DOTALL)
_JSON_END_RE = re.compile(r"\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match_start = _JSON_START_RE.search(text)
    match_end = _JSON_END_RE.search(text)
    if not match_start or not match_end:
        return None
    try:
        return json.loads(text[match_start.start() : match_end.end()])
    except json.JSONDecodeError:
        return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("°", "")
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            return float(match.group(0))
    return None


def _usage_token_count(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    return int(value) if value is not None else None


def _estimate_cost(
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
    input_price_per_million: float,
    output_price_per_million: float,
) -> Optional[float]:
    if prompt_tokens is None or output_tokens is None:
        return None
    cost = (prompt_tokens / 1_000_000) * input_price_per_million
    cost += (output_tokens / 1_000_000) * output_price_per_million
    return cost


def read_gemini_ocr(
    image_path: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    input_price_per_million: float = DEFAULT_INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = DEFAULT_OUTPUT_PRICE_PER_MILLION,
) -> GeminiOCRResult:
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Install it with `pip install google-generativeai`."
        ) from _GENAI_IMPORT_ERROR

    genai.configure(api_key=api_key)
    image = Image.open(image_path)

    prompt = (
        "You are reading a digital pressure/temperature display from an HRC-110. "
        "Return JSON only with keys: pressure, temperature, pressure_unit, temperature_unit. "
        "Use null for values you cannot read. Pressure is typically in MPa and temperature in °C."
    )

    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(
        [prompt, image],
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=200,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text or ""
    payload = _extract_json(raw_text) or {}
    pressure = _parse_float(payload.get("pressure"))
    temperature = _parse_float(payload.get("temperature"))

    pressure_unit = str(payload.get("pressure_unit") or "MPa")
    temperature_unit = str(payload.get("temperature_unit") or "C")

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = _usage_token_count(usage, "prompt_token_count")
    output_tokens = _usage_token_count(usage, "candidates_token_count")
    estimated_cost = _estimate_cost(
        prompt_tokens,
        output_tokens,
        input_price_per_million,
        output_price_per_million,
    )

    return GeminiOCRResult(
        pressure=pressure,
        temperature=temperature,
        pressure_unit=pressure_unit,
        temperature_unit=temperature_unit,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost,
        raw_response=raw_text,
    )


def _format_result(image_path: Path, result: GeminiOCRResult) -> str:
    pressure = (
        f"{result.pressure:.2f} {result.pressure_unit}" if result.pressure is not None else "n/a"
    )
    temperature = (
        f"{result.temperature:.3f} {result.temperature_unit}"
        if result.temperature is not None
        else "n/a"
    )
    tokens = (
        f"prompt={result.prompt_tokens}, output={result.output_tokens}"
        if result.prompt_tokens is not None
        else "token usage n/a"
    )
    cost = (
        f"${result.estimated_cost_usd:.6f}"
        if result.estimated_cost_usd is not None
        else "n/a"
    )
    return (
        f"{image_path.name}: pressure={pressure}, temperature={temperature}, "
        f"cost={cost}, {tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemini OCR on a single image or a directory of images.",
    )
    parser.add_argument("image", type=Path, help="Image path or directory of images")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE_PER_MILLION)
    parser.add_argument("--output-price", type=float, default=DEFAULT_OUTPUT_PRICE_PER_MILLION)

    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    image_path = args.image
    if image_path.is_dir():
        images = sorted(
            path
            for path in image_path.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        images = [image_path]

    for path in images:
        result = read_gemini_ocr(
            path,
            api_key=args.api_key,
            model=args.model,
            input_price_per_million=args.input_price,
            output_price_per_million=args.output_price,
        )
        print(_format_result(path, result))


if __name__ == "__main__":
    main()
