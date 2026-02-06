# HRC-100/110 Pressure Monitor (CLI Prototype)

This repository contains a **minimal Python CLI** intended to validate USB connectivity and basic serial communication with a **Helium Recondenser Controller HRC‑100/110** before building the full monitoring/alarm application.

> **Note on documentation & research:** The environment used to assemble this first step does not allow outbound network access or PDF text extraction tools. The manuals are present in `docs/`, but I cannot extract or search them here. The CLI is therefore configurable for baud rate, parity, stop bits, and payload so you can quickly test and refine parameters once you review the vendor docs locally. See the **Next steps** section for how we’ll fold in the protocol details once you confirm them.

## Background & Goal

You asked for an application similar to [`SCM10_temperature_monitor`](https://github.com/MagShadow/SCM10_temperature_monitor) but targeting the **HRC‑100/110 pressure meter**, capable of **monitoring pressure** and **sending alarms**. This first milestone is a **minimal USB-only CLI** that:

- Detects USB serial devices.
- Opens a selected port with configurable serial parameters.
- Optionally sends a hex or ASCII payload.
- Reads back the response so we can confirm communication.

Once communication is verified, we will:

1. Implement the HRC‑100/110 protocol (commands, registers, checksum, etc.).
2. Add periodic polling for pressure values.
3. Implement alarm rules and notifications.
4. Package as a service/daemon and add logging.

## Quick Start

### 1) Create a conda environment and install dependencies

```bash
conda env create -f environment.yml
conda activate hrc110
```

If you need to install/update dependencies manually inside the environment:

```bash
pip install -r requirements.txt
```

### 2) List USB serial devices

```bash
python hrc110_cli.py list
```

### 3) Probe a port (USB only)

```bash
python hrc110_cli.py probe --port /dev/ttyUSB0 --baud 9600 --parity N --stopbits 1 --timeout 1.0
```

### 4) Send a hex payload (example)

```bash
python hrc110_cli.py probe \
  --port /dev/ttyUSB0 \
  --baud 9600 \
  --write-hex "01 03 00 00 00 02 C4 0B" \
  --read-bytes 64
```

If you prefer ASCII commands:

```bash
python hrc110_cli.py probe \
  --port /dev/ttyUSB0 \
  --write-ascii "STATUS?\r\n" \
  --read-bytes 128
```

## Next Steps (After You Test)

Once you confirm the correct port settings and any working command examples from the HRC‑100/110 manual:

- I will encode the **official protocol** (queries, scaling, unit conversion, checksums).
- We will add **pressure polling** at a configured interval.
- We will add **alarm configuration** (thresholds, hysteresis, notifications).

Please share the confirmed **baud rate**, **parity**, **stop bits**, and any **known query commands** (or register map) from the manual so we can implement the proper protocol in the next iteration.

## Files

- `hrc110_cli.py` — Minimal USB serial CLI for connection testing.
- `requirements.txt` — Python dependencies.
- `environment.yml` — Conda environment definition.
- `docs/` — Vendor documentation (manuals/appendices).
