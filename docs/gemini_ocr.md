# Gemini OCR (experimental)

This is an alternative OCR path that uses Google Gemini's vision model. It is **not** wired
into the main application yet, so you can experiment without impacting the current OCR flow.

## Quick start

1. Install the optional dependency:

   ```bash
   pip install google-generativeai
   ```

2. Create an API key and export it:

   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```

3. Run OCR on a single image:

   ```bash
   python pressure_ocr_gemini.py tests_photos/WIN_20231108_20_46_55_Pro.jpg
   ```

4. Run the batch test helper against the existing `tests_photos` directory:

   ```bash
   python gemini_ocr_test.py --tests-dir tests_photos
   ```

## API key setup

- Go to Google AI Studio and create a Gemini API key.
- Store the key as `GEMINI_API_KEY` in your shell or in a local `.env` file.
- The scripts read `GEMINI_API_KEY` by default, or you can pass `--api-key` explicitly.

## Cost calculation

The Gemini API reports token usage in the response metadata. The Gemini OCR script uses that
usage to estimate per-request cost:

```
estimated_cost_usd = (prompt_tokens / 1,000,000) * input_price
                   + (output_tokens / 1,000,000) * output_price
```

Defaults are set for Gemini 2.5 Flash Lite (input: $0.35 / 1M tokens, output: $1.05 / 1M tokens),
which is currently the cheapest generally available vision model. Update the prices with
`--input-price` and `--output-price` to match the latest pricing or a different model.

### Image-size based cost estimate (fallback)

If the API response does not include token usage, the script estimates prompt tokens from the
image size. It uses a 512×512 tile size and 258 tokens per tile:

```
tiles_w = ceil(width / 512)
tiles_h = ceil(height / 512)
prompt_tokens ≈ tiles_w * tiles_h * 258
```

This is an approximation; for exact costs, prefer the token usage returned by the API.
