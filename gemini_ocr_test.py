from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from pressure_ocr_gemini import (
    DEFAULT_INPUT_PRICE_PER_MILLION,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_PRICE_PER_MILLION,
    read_gemini_ocr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini OCR over tests_photos.")
    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests_photos"),
        help="Directory containing test photos.",
    )
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("data/gemini_ocr_results.csv"))
    parser.add_argument("--input-price", type=float, default=None)
    parser.add_argument("--output-price", type=float, default=None)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set GEMINI_API_KEY or pass --api-key.")

    images = sorted(
        path
        for path in args.tests_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No images found in {args.tests_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "pressure",
                "temperature",
                "pressure_unit",
                "temperature_unit",
                "prompt_tokens",
                "output_tokens",
                "estimated_cost_usd",
            ]
        )

        for image in images:
            result = read_gemini_ocr(
                image,
                api_key=args.api_key,
                model=args.model,
                input_price_per_million=args.input_price or DEFAULT_INPUT_PRICE_PER_MILLION,
                output_price_per_million=args.output_price or DEFAULT_OUTPUT_PRICE_PER_MILLION,
            )
            writer.writerow(
                [
                    image.name,
                    result.pressure,
                    result.temperature,
                    result.pressure_unit,
                    result.temperature_unit,
                    result.prompt_tokens,
                    result.output_tokens,
                    result.estimated_cost_usd,
                ]
            )
            print(f"{image.name}: cost=${result.estimated_cost_usd or 0:.6f}")

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
