#!/usr/bin/env python3
"""Minimal USB serial CLI for HRC-100/110 connection testing."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import serial
from serial.tools import list_ports


def parse_hex_payload(payload: str) -> bytes:
    cleaned = payload.replace("0x", " ").replace(",", " ").replace(";", " ")
    chunks = [chunk for chunk in cleaned.split() if chunk]
    try:
        return bytes(int(chunk, 16) for chunk in chunks)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid hex payload. Use space-separated bytes, e.g. '01 03 00 00 00 02 C4 0B'."
        ) from exc


def format_hex(data: bytes) -> str:
    return " ".join(f"{byte:02X}" for byte in data)


@dataclass
class PortInfo:
    device: str
    description: str
    vid: Optional[int]
    pid: Optional[int]
    serial_number: Optional[str]
    location: Optional[str]


def iter_usb_ports(usb_only: bool = True) -> Iterable[PortInfo]:
    for port in list_ports.comports():
        if usb_only and port.vid is None:
            continue
        yield PortInfo(
            device=port.device,
            description=port.description,
            vid=port.vid,
            pid=port.pid,
            serial_number=port.serial_number,
            location=port.location,
        )


def list_ports_cmd(args: argparse.Namespace) -> int:
    ports = list(iter_usb_ports(usb_only=not args.include_non_usb))
    if not ports:
        scope = "USB" if not args.include_non_usb else "serial"
        print(f"No {scope} serial devices found.")
        return 1

    for port in ports:
        vid_pid = (
            f"{port.vid:04X}:{port.pid:04X}" if port.vid is not None else "N/A"
        )
        details = [port.description, f"VID:PID={vid_pid}"]
        if port.serial_number:
            details.append(f"SN={port.serial_number}")
        if port.location:
            details.append(f"LOC={port.location}")
        print(f"{port.device} :: " + " | ".join(details))
    return 0


def open_serial(args: argparse.Namespace) -> serial.Serial:
    try:
        return serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=args.bytesize,
            parity=args.parity,
            stopbits=args.stopbits,
            timeout=args.timeout,
        )
    except serial.SerialException as exc:
        raise SystemExit(f"Failed to open {args.port}: {exc}") from exc


def probe_cmd(args: argparse.Namespace) -> int:
    payload = b""
    if args.write_hex:
        payload = parse_hex_payload(args.write_hex)
    elif args.write_ascii:
        payload = args.write_ascii.encode("ascii")

    with open_serial(args) as connection:
        if args.toggle_dtr:
            connection.dtr = False
            connection.dtr = True
        if args.toggle_rts:
            connection.rts = False
            connection.rts = True

        if payload:
            connection.reset_input_buffer()
            connection.reset_output_buffer()
            written = connection.write(payload)
            connection.flush()
            print(f"Wrote {written} bytes: {format_hex(payload)}")
        else:
            print("No payload written. Listening for data...")

        response = connection.read(args.read_bytes)
        if response:
            print(f"Read {len(response)} bytes: {format_hex(response)}")
        else:
            print("No response received.")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HRC-100/110 USB serial connection test CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List serial ports.")
    list_parser.add_argument(
        "--include-non-usb",
        action="store_true",
        help="Include non-USB serial devices.",
    )
    list_parser.set_defaults(func=list_ports_cmd)

    probe_parser = subparsers.add_parser(
        "probe", help="Open a serial port and optionally send data."
    )
    probe_parser.add_argument("--port", required=True, help="Serial port device path.")
    probe_parser.add_argument("--baud", type=int, default=9600, help="Baud rate.")
    probe_parser.add_argument(
        "--bytesize",
        type=int,
        default=8,
        choices=[5, 6, 7, 8],
        help="Data bits.",
    )
    probe_parser.add_argument(
        "--parity",
        default="N",
        choices=["N", "E", "O", "M", "S"],
        help="Parity (N/E/O/M/S).",
    )
    probe_parser.add_argument(
        "--stopbits",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stop bits.",
    )
    probe_parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Read timeout in seconds.",
    )
    probe_parser.add_argument(
        "--write-hex",
        help="Hex payload to write (e.g. '01 03 00 00 00 02 C4 0B').",
    )
    probe_parser.add_argument(
        "--write-ascii",
        help="ASCII payload to write (e.g. 'STATUS?\\r\\n').",
    )
    probe_parser.add_argument(
        "--read-bytes",
        type=int,
        default=64,
        help="Number of bytes to read after writing.",
    )
    probe_parser.add_argument(
        "--toggle-dtr",
        action="store_true",
        help="Toggle DTR line before communication.",
    )
    probe_parser.add_argument(
        "--toggle-rts",
        action="store_true",
        help="Toggle RTS line before communication.",
    )
    probe_parser.set_defaults(func=probe_cmd)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
