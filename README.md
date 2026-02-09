# HRC-100/110 Pressure Monitor (GUI + OCR)

This repository provides a **GUI pressure monitor** modeled after the layout of the SCM10 temperature monitor, but tailored to **HRC‑100/110** pressure readings captured by a **USB camera** and decoded via OCR.

## Features

- **SCM10-style layout**
  - Left upper: camera selection + test OCR
  - Left lower: alarm thresholds + email setup
  - Right: live pressure reading, status, and plot
- **USB camera capture** → **OCR** → **record + plot**
- **Status updates** (capture time, OCR progress)
- **Minimum interval enforced (2 minutes)**
- **Temporary photos auto-deleted after OCR**
- **OCR error handling** with pop-up notifications (skips bad points)

## Quick Start

### 1) Create a conda environment and install dependencies

```bash
conda env create -f environment.yml
conda activate hrc110
```

Or install with pip:

```bash
pip install -r requirements.txt
```

### 2) Launch the GUI monitor

```bash
python pressure_monitor_gui.py
```

## GUI Usage Notes

- **Camera test** captures a photo and runs OCR immediately.
- **Test captures** are saved to `data/last_capture.jpg` so you can inspect failed OCR images.
- **Monitoring interval** is in minutes (minimum 2).
- **Data logging**: readings are appended to `data/pressure_readings.csv`.
- **Alarms**: set low/high thresholds and SMTP details to receive alert emails.
- **OCR failures** trigger a pop-up and skip the current data point.

## OCR Prototype CLI

You can still run OCR directly on a photo:

```bash
python pressure_ocr.py --image test_photo.jpg
```

## Experimental Gemini OCR (not wired into the GUI yet)

An optional Gemini-based OCR prototype is available for comparison. See
`docs/gemini_ocr.md` for setup, batch testing, and per-request cost calculation.

## Files

- `pressure_monitor_gui.py` — GUI app for monitoring pressure and sending alarms.
- `pressure_ocr.py` — OCR pipeline for reading pressure/temperature from a photo.
- `pressure_ocr_gemini.py` — Experimental Gemini-based OCR (not integrated yet).
- `gemini_ocr_test.py` — Batch test helper for `tests_photos`.
- `hrc110_cli.py` — USB serial CLI for connection testing.
- `requirements.txt` — Python dependencies.
- `environment.yml` — Conda environment definition.
- `docs/` — Vendor documentation (manuals/appendices).
