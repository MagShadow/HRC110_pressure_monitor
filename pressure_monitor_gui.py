#!/usr/bin/env python3
"""GUI monitor for HRC-100/110 pressure OCR readings."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import smtplib
import ssl
import tempfile
import tkinter as tk
from dataclasses import dataclass
from email.message import EmailMessage
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pressure_ocr import read_pressure_temperature
try:
    from pressure_ocr_gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
    from pressure_ocr_gemini import read_gemini_ocr
except Exception as exc:  # pragma: no cover - optional runtime dependency
    read_gemini_ocr = None
    GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"
    _GEMINI_IMPORT_ERROR = exc
else:
    _GEMINI_IMPORT_ERROR = None

MIN_INTERVAL_MINUTES = 1
DEFAULT_INTERVAL_MINUTES = 10
DATA_DIR = "data"
CSV_NAME = "pressure_readings.csv"
STATUS_LOG_NAME = "pressure_status.log"
CONFIG_NAME = "monitor_config.json"
OCR_METHOD_LOCAL = "Local"
OCR_METHOD_GEMINI = "GEMINI"
OCR_METHODS = (OCR_METHOD_LOCAL, OCR_METHOD_GEMINI)
BG_COLOR = "#ECECEA"
SECTION_BG = "#E5E4E0"
STATUS_BG = "#FFFFFF"
STATUS_FG = "#1F1F1F"
ALERT_COLOR = "#B5322E"


@dataclass
class AlarmConfig:
    enabled: bool
    beep_enabled: bool
    email_enabled: bool
    low_threshold: float | None
    high_threshold: float | None
    recipient: str
    sender: str
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool


class PressureMonitorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HRC-110 Pressure Monitor")
        self.root.configure(bg=BG_COLOR)

        self.monitoring = False
        self.after_id: str | None = None
        self.timestamps: list[dt.datetime] = []
        self.values: list[float] = []
        self.last_alarm_active = False
        self.last_alarm_sent: dt.datetime | None = None
        self.gemini_total_cost_usd = 0.0
        self.reading_log_path = os.path.join(DATA_DIR, CSV_NAME)
        self.status_log_path = os.path.join(DATA_DIR, STATUS_LOG_NAME)
        self.config_path = os.path.join(DATA_DIR, CONFIG_NAME)

        self._configure_styles()
        self._build_layout()
        self._refresh_cameras()
        self._load_config_from_path(self.config_path, silent=True)
        self._update_gemini_cost_label()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        self.root.option_add("*Font", "{Segoe UI} 10")
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground="#202020")
        style.configure("TButton", padding=(8, 4))
        style.configure("TLabelframe", background=SECTION_BG, borderwidth=1, relief="solid")
        style.configure(
            "TLabelframe.Label",
            background=SECTION_BG,
            foreground="#202020",
            font=("{Segoe UI}", 10, "bold"),
        )
        style.configure("Value.TLabel", font=("{Segoe UI}", 30, "bold"), foreground=ALERT_COLOR)
        style.configure("Secondary.TLabel", foreground="#3A3A3A")

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_camera_panel(left)
        self._build_alarm_panel(left)
        self._build_reading_panel(right)

    def _build_camera_panel(self, parent: ttk.Frame) -> None:
        camera_frame = ttk.LabelFrame(parent, text="Camera", padding=10)
        camera_frame.grid(row=0, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        camera_frame.columnconfigure(1, weight=1)
        camera_frame.columnconfigure(2, weight=1)

        ttk.Label(camera_frame, text="Select camera index:").grid(
            row=0, column=0, sticky="w"
        )
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            camera_frame, textvariable=self.camera_var, state="readonly", width=12
        )
        self.camera_combo.grid(row=0, column=1, padx=(5, 0))
        ttk.Button(
            camera_frame, text="Refresh", command=self._refresh_cameras
        ).grid(row=0, column=2, padx=(5, 0))

        ttk.Label(camera_frame, text="OCR method:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ocr_method_var = tk.StringVar(value=OCR_METHOD_LOCAL)
        self.ocr_method_combo = ttk.Combobox(
            camera_frame,
            textvariable=self.ocr_method_var,
            values=OCR_METHODS,
            state="readonly",
            width=12,
        )
        self.ocr_method_combo.grid(row=1, column=1, padx=(5, 0), pady=(8, 0), sticky="w")
        self.ocr_method_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_gemini_cost_label())

        ttk.Label(camera_frame, text="Gemini API key:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.gemini_api_key_var = tk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))
        ttk.Entry(
            camera_frame, textvariable=self.gemini_api_key_var, width=28, show="*"
        ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(8, 0))

        test_row = ttk.Frame(camera_frame)
        test_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        test_row.columnconfigure(0, weight=1)
        test_row.columnconfigure(1, weight=1)
        ttk.Button(test_row, text="Test Camera", command=self._test_camera).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(
            test_row, text="Test OCR", command=self._test_ocr
        ).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self.camera_status = ttk.Label(camera_frame, text="")
        self.camera_status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_alarm_panel(self, parent: ttk.Frame) -> None:
        alarm_frame = ttk.LabelFrame(parent, text="Alarm & Email", padding=10)
        alarm_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        parent.rowconfigure(1, weight=1)
        alarm_frame.columnconfigure(1, weight=1)

        ttk.Label(alarm_frame, text="Reading log file:").grid(row=0, column=0, sticky="w")
        self.reading_log_var = tk.StringVar(value=self.reading_log_path)
        ttk.Entry(alarm_frame, textvariable=self.reading_log_var, width=24).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(alarm_frame, text="Browse", command=self._browse_reading_log_file).grid(
            row=0, column=2, padx=(6, 0)
        )

        ttk.Label(alarm_frame, text="Status log file:").grid(
            row=1, column=0, sticky="w"
        )
        self.status_log_var = tk.StringVar(value=self.status_log_path)
        ttk.Entry(alarm_frame, textvariable=self.status_log_var, width=24).grid(
            row=1, column=1, sticky="ew", pady=(4, 0)
        )
        ttk.Button(alarm_frame, text="Browse", command=self._browse_status_log_file).grid(
            row=1, column=2, padx=(6, 0), pady=(4, 0)
        )

        ttk.Label(alarm_frame, text="Interval (minutes):").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.interval_var = tk.StringVar(value=str(DEFAULT_INTERVAL_MINUTES))
        ttk.Entry(alarm_frame, textvariable=self.interval_var, width=10).grid(
            row=2, column=1, sticky="w", pady=(8, 0)
        )
        ttk.Label(alarm_frame, text=f"(min {MIN_INTERVAL_MINUTES})").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )

        self.alarm_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            alarm_frame, text="Enable alarm", variable=self.alarm_enabled_var
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.beep_alarm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            alarm_frame, text="Beep alarm", variable=self.beep_alarm_var
        ).grid(row=4, column=0, sticky="w")

        self.email_alarm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            alarm_frame, text="Email alarm", variable=self.email_alarm_var
        ).grid(row=4, column=1, sticky="w")

        ttk.Label(alarm_frame, text="Low threshold:").grid(row=5, column=0, sticky="w")
        self.low_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.low_var, width=10).grid(
            row=5, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="High threshold:").grid(row=6, column=0, sticky="w")
        self.high_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.high_var, width=10).grid(
            row=6, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="Email recipient:").grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        self.recipient_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.recipient_var, width=24).grid(
            row=7, column=1, columnspan=2, sticky="ew", pady=(8, 0)
        )

        ttk.Label(alarm_frame, text="Email sender:").grid(row=8, column=0, sticky="w")
        self.sender_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.sender_var, width=24).grid(
            row=8, column=1, columnspan=2, sticky="ew"
        )

        ttk.Label(alarm_frame, text="SMTP server:").grid(row=9, column=0, sticky="w")
        self.smtp_server_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_server_var, width=24).grid(
            row=9, column=1, columnspan=2, sticky="ew"
        )

        ttk.Label(alarm_frame, text="SMTP port:").grid(row=10, column=0, sticky="w")
        self.smtp_port_var = tk.StringVar(value="587")
        ttk.Entry(alarm_frame, textvariable=self.smtp_port_var, width=10).grid(
            row=10, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="SMTP user:").grid(row=11, column=0, sticky="w")
        self.smtp_user_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_user_var, width=24).grid(
            row=11, column=1, columnspan=2, sticky="ew"
        )

        ttk.Label(alarm_frame, text="SMTP password:").grid(row=12, column=0, sticky="w")
        self.smtp_pass_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_pass_var, width=24, show="*").grid(
            row=12, column=1, columnspan=2, sticky="ew"
        )

        self.smtp_tls_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            alarm_frame, text="Use TLS", variable=self.smtp_tls_var
        ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(4, 0))

        config_row = ttk.Frame(alarm_frame)
        config_row.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(config_row, text="Load Config", command=self._load_config).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(config_row, text="Save Config", command=self._save_config).grid(
            row=0, column=1
        )

        button_row = ttk.Frame(alarm_frame)
        button_row.grid(row=15, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(button_row, text="Start", command=self._start_monitoring).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(button_row, text="Stop", command=self._stop_monitoring).grid(
            row=0, column=1
        )

    def _build_reading_panel(self, parent: ttk.Frame) -> None:
        reading_frame = ttk.LabelFrame(parent, text="Pressure Reading", padding=10)
        reading_frame.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        reading_frame.columnconfigure(0, weight=1)
        reading_frame.rowconfigure(3, weight=1)

        self.current_value_label = ttk.Label(
            reading_frame, text="--", style="Value.TLabel"
        )
        self.current_value_label.grid(row=0, column=0, sticky="w")

        self.status_label = ttk.Label(reading_frame, text="Idle", style="Secondary.TLabel")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(6, 4))

        self.gemini_cost_label = ttk.Label(
            reading_frame, text="Gemini total cost (since start): $0.000000"
        )
        self.gemini_cost_label.grid(row=2, column=0, sticky="w", pady=(0, 4))

        self.figure = Figure(figsize=(6, 3), dpi=100, facecolor=SECTION_BG)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor("#FFFFFF")
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Pressure")
        self.axis.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=reading_frame)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=(4, 8))

        status_frame = ttk.LabelFrame(reading_frame, text="Status Log")
        status_frame.grid(row=4, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.status_text = tk.Text(status_frame, height=6, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")
        self.status_text.configure(
            bg=STATUS_BG,
            fg=STATUS_FG,
            insertbackground=STATUS_FG,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10),
        )

    def _refresh_cameras(self) -> None:
        available = []
        for index in range(6):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                available.append(str(index))
            capture.release()

        if not available:
            available = ["0"]
            self.camera_status.configure(text="No camera detected. Defaulting to 0.")
        else:
            self.camera_status.configure(text=f"Detected cameras: {', '.join(available)}")

        self.camera_combo["values"] = available
        if self.camera_var.get() not in available:
            self.camera_var.set(available[0])

    def _browse_reading_log_file(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select Reading Log File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=os.path.basename(self.reading_log_var.get() or CSV_NAME),
        )
        if selected:
            self.reading_log_var.set(selected)

    def _browse_status_log_file(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Select Status Log File",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=os.path.basename(self.status_log_var.get() or STATUS_LOG_NAME),
        )
        if selected:
            self.status_log_var.set(selected)

    def _save_config(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save Monitor Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=os.path.basename(self.config_path),
        )
        if not selected:
            return
        payload = self._build_config_payload()
        self._ensure_parent_dir(selected)
        with open(selected, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.config_path = selected
        self._log_status("Config saved (includes Gemini API key in plain text).")

    def _load_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load Monitor Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not selected:
            return
        self._load_config_from_path(selected, silent=False)

    def _load_config_from_path(self, path: str, silent: bool) -> None:
        if not os.path.exists(path):
            if not silent:
                messagebox.showerror("Config error", f"Config file not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._apply_config_payload(payload)
            self.config_path = path
            if not silent:
                self._log_status(f"Config loaded from {path}")
        except Exception as exc:
            if not silent:
                messagebox.showerror("Config error", f"Failed to load config:\n{exc}")

    def _build_config_payload(self) -> dict[str, object]:
        return {
            "camera_index": self.camera_var.get().strip(),
            "ocr_method": self.ocr_method_var.get().strip(),
            "gemini_api_key": self.gemini_api_key_var.get(),
            "reading_log_file": self.reading_log_var.get().strip(),
            "status_log_file": self.status_log_var.get().strip(),
            "interval_minutes": self.interval_var.get().strip(),
            "alarm_enabled": self.alarm_enabled_var.get(),
            "beep_alarm_enabled": self.beep_alarm_var.get(),
            "email_alarm_enabled": self.email_alarm_var.get(),
            "low_threshold": self.low_var.get().strip(),
            "high_threshold": self.high_var.get().strip(),
            "recipient": self.recipient_var.get().strip(),
            "sender": self.sender_var.get().strip(),
            "smtp_server": self.smtp_server_var.get().strip(),
            "smtp_port": self.smtp_port_var.get().strip(),
            "smtp_username": self.smtp_user_var.get().strip(),
            "smtp_password": self.smtp_pass_var.get(),
            "smtp_use_tls": self.smtp_tls_var.get(),
        }

    def _apply_config_payload(self, payload: dict[str, object]) -> None:
        camera_index = str(payload.get("camera_index", self.camera_var.get()))
        if camera_index:
            self.camera_var.set(camera_index)
        ocr_method = str(payload.get("ocr_method", self.ocr_method_var.get()))
        if ocr_method in OCR_METHODS:
            self.ocr_method_var.set(ocr_method)
        self.gemini_api_key_var.set(str(payload.get("gemini_api_key", self.gemini_api_key_var.get())))
        self.reading_log_var.set(str(payload.get("reading_log_file", self.reading_log_var.get())))
        self.status_log_var.set(str(payload.get("status_log_file", self.status_log_var.get())))
        self.interval_var.set(str(payload.get("interval_minutes", self.interval_var.get())))
        self.alarm_enabled_var.set(bool(payload.get("alarm_enabled", self.alarm_enabled_var.get())))
        self.beep_alarm_var.set(bool(payload.get("beep_alarm_enabled", self.beep_alarm_var.get())))
        self.email_alarm_var.set(bool(payload.get("email_alarm_enabled", self.email_alarm_var.get())))
        self.low_var.set(str(payload.get("low_threshold", self.low_var.get())))
        self.high_var.set(str(payload.get("high_threshold", self.high_var.get())))
        self.recipient_var.set(str(payload.get("recipient", self.recipient_var.get())))
        self.sender_var.set(str(payload.get("sender", self.sender_var.get())))
        self.smtp_server_var.set(str(payload.get("smtp_server", self.smtp_server_var.get())))
        self.smtp_port_var.set(str(payload.get("smtp_port", self.smtp_port_var.get())))
        self.smtp_user_var.set(str(payload.get("smtp_username", self.smtp_user_var.get())))
        self.smtp_pass_var.set(str(payload.get("smtp_password", self.smtp_pass_var.get())))
        self.smtp_tls_var.set(bool(payload.get("smtp_use_tls", self.smtp_tls_var.get())))

    def _test_camera(self) -> None:
        self._log_status("Testing camera capture...")
        self._update_status("Testing camera")
        try:
            image_path, temp_path = self._capture_photo(keep_photo=True)
        except Exception as exc:
            messagebox.showerror("Camera error", str(exc))
            self._update_status("Camera test failed")
            return
        finally:
            if "temp_path" in locals() and temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        self._update_status("Camera test passed")
        self._log_status(f"Camera test capture saved to {image_path}")
        messagebox.showinfo("Camera test", f"Capture successful.\nSaved to {image_path}.")

    def _test_ocr(self) -> None:
        method = self._selected_ocr_method()
        self._log_status(f"Testing OCR ({method})...")
        self._update_status("Testing OCR")
        try:
            reading, request_cost = self._capture_and_ocr(keep_photo=True)
        except Exception as exc:
            messagebox.showerror(
                "OCR test error",
                f"{exc}\n\nLast capture saved to data/last_capture.jpg (if available).",
            )
            self._update_status("OCR test failed")
            return
        self._update_status(f"OCR test OK: {reading:.2f}")
        if method == OCR_METHOD_GEMINI and request_cost is not None:
            messagebox.showinfo(
                "OCR test",
                f"Pressure reading: {reading:.2f}\nEstimated request cost: ${request_cost:.6f}",
            )
        else:
            messagebox.showinfo("OCR test", f"Pressure reading: {reading:.2f}")

    def _start_monitoring(self) -> None:
        interval = self._interval_minutes()
        if interval is None:
            return
        if interval < MIN_INTERVAL_MINUTES:
            messagebox.showerror(
                "Invalid interval",
                f"Minimum interval is {MIN_INTERVAL_MINUTES} minutes.",
            )
            return

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        self._ensure_parent_dir(self.reading_log_var.get().strip() or self.reading_log_path)
        self._ensure_parent_dir(self.status_log_var.get().strip() or self.status_log_path)
        self.gemini_total_cost_usd = 0.0
        self._update_gemini_cost_label()

        self.monitoring = True
        self._update_status("Monitoring started")
        self._schedule_next_run(initial=True)

    def _stop_monitoring(self) -> None:
        self.monitoring = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self._update_status("Monitoring stopped")

    def _interval_minutes(self) -> int | None:
        try:
            return int(self.interval_var.get())
        except ValueError:
            messagebox.showerror("Invalid interval", "Interval must be an integer.")
            return None

    def _schedule_next_run(self, initial: bool = False) -> None:
        if not self.monitoring:
            return
        interval = self._interval_minutes()
        if interval is None:
            return
        interval = max(MIN_INTERVAL_MINUTES, interval)

        delay_ms = 1000 if initial else interval * 60 * 1000
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(delay_ms, self._run_cycle)

    def _run_cycle(self) -> None:
        if not self.monitoring:
            return
        self._update_status("Capturing photo")
        try:
            value, request_cost = self._capture_and_ocr()
        except Exception as exc:
            self._update_status("OCR error")
            messagebox.showerror("OCR error", str(exc))
            self._log_status(f"OCR error: {exc}")
            self._schedule_next_run()
            return

        if value is not None:
            self._record_value(value)
            if self._selected_ocr_method() == OCR_METHOD_GEMINI and request_cost is not None:
                self.gemini_total_cost_usd += request_cost
                self._update_gemini_cost_label()
            self._check_alarm(value)
        self._schedule_next_run()

    def _selected_ocr_method(self) -> str:
        method = self.ocr_method_var.get().strip()
        if method not in OCR_METHODS:
            return OCR_METHOD_LOCAL
        return method

    def _capture_photo(self, keep_photo: bool = False) -> tuple[str, str | None]:
        camera_index = int(self.camera_var.get())
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")
        success, frame = capture.read()
        capture.release()
        if not success:
            raise RuntimeError("Unable to capture photo from camera")

        timestamp = dt.datetime.now()
        self._log_status(f"Photo captured at {timestamp:%Y-%m-%d %H:%M:%S}")

        temp_path = None
        if keep_photo:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            image_path = os.path.join(DATA_DIR, "last_capture.jpg")
            cv2.imwrite(image_path, frame)
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
            image_path = temp_path
        return image_path, temp_path

    def _capture_and_ocr(self, keep_photo: bool = False) -> tuple[float, float | None]:
        image_path, temp_path = self._capture_photo(keep_photo=keep_photo)

        self._update_status("Running OCR")
        request_cost_usd: float | None = None
        try:
            method = self._selected_ocr_method()
            if method == OCR_METHOD_LOCAL:
                pressure, _temperature = read_pressure_temperature(image_path)
                value = float(pressure.value)
                self._log_status(
                    f"Local OCR result: {pressure.value} (confidence={pressure.confidence:.2f})"
                )
            elif method == OCR_METHOD_GEMINI:
                if read_gemini_ocr is None:
                    raise RuntimeError(
                        "Gemini OCR dependencies are unavailable: "
                        f"{_GEMINI_IMPORT_ERROR}"
                    )
                api_key = self.gemini_api_key_var.get().strip() or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "Gemini API key is missing. Enter it in the GUI field or set GEMINI_API_KEY."
                    )
                result = read_gemini_ocr(
                    image_path=image_path,
                    api_key=api_key,
                    model=GEMINI_DEFAULT_MODEL,
                )
                if result.pressure is None:
                    raise RuntimeError("Gemini OCR returned no pressure value.")
                value = float(result.pressure)
                request_cost_usd = result.estimated_cost_usd
                self._log_status(
                    f"Gemini OCR result: {value:.2f} "
                    f"(prompt_tokens={result.prompt_tokens}, output_tokens={result.output_tokens})"
                )
            else:
                raise RuntimeError(f"Unsupported OCR method: {method}")
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if keep_photo:
            self._log_status(f"Saved capture to {image_path}")

        return value, request_cost_usd

    def _record_value(self, value: float) -> None:
        now = dt.datetime.now()
        self.timestamps.append(now)
        self.values.append(value)
        self.current_value_label.configure(text=f"{value:.2f}")
        self._update_plot()
        self._append_csv(now, value)
        self._update_status("Reading captured")

    def _append_csv(self, timestamp: dt.datetime, value: float) -> None:
        path = self.reading_log_var.get().strip() or self.reading_log_path
        self._ensure_parent_dir(path)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["timestamp", "pressure"])
            writer.writerow([timestamp.isoformat(), f"{value:.2f}"])

    def _update_plot(self) -> None:
        self.axis.clear()
        self.axis.grid(True)
        if self.timestamps:
            self.axis.plot(self.timestamps, self.values, marker="o", linestyle="-")
            self.axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            self.figure.autofmt_xdate(rotation=25, ha="right")
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Pressure")
        self.canvas.draw()

    def _check_alarm(self, value: float) -> None:
        try:
            config = self._alarm_config()
        except ValueError as exc:
            messagebox.showerror("Invalid alarm settings", str(exc))
            self._log_status(f"Alarm config error: {exc}")
            return
        if not config.enabled:
            self.last_alarm_active = False
            return

        outside_low = config.low_threshold is not None and value < config.low_threshold
        outside_high = config.high_threshold is not None and value > config.high_threshold
        alarm_active = outside_low or outside_high

        if alarm_active and not self.last_alarm_active:
            if config.beep_enabled:
                self.root.bell()
                self._log_status(f"Beep alarm triggered at pressure={value:.2f}")
            if config.email_enabled:
                self._send_alarm_email(config, value)
            if not config.beep_enabled and not config.email_enabled:
                self._log_status("Alarm triggered but both beep/email alarms are disabled.")
        self.last_alarm_active = alarm_active

    def _alarm_config(self) -> AlarmConfig:
        def parse_optional(value: str) -> float | None:
            value = value.strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid threshold value: {value}") from exc

        low = parse_optional(self.low_var.get())
        high = parse_optional(self.high_var.get())
        port_raw = self.smtp_port_var.get().strip()
        try:
            smtp_port = int(port_raw) if port_raw else 0
        except ValueError as exc:
            raise ValueError(f"Invalid SMTP port value: {port_raw}") from exc
        return AlarmConfig(
            enabled=self.alarm_enabled_var.get(),
            beep_enabled=self.beep_alarm_var.get(),
            email_enabled=self.email_alarm_var.get(),
            low_threshold=low,
            high_threshold=high,
            recipient=self.recipient_var.get().strip(),
            sender=self.sender_var.get().strip(),
            smtp_server=self.smtp_server_var.get().strip(),
            smtp_port=smtp_port,
            username=self.smtp_user_var.get().strip(),
            password=self.smtp_pass_var.get().strip(),
            use_tls=self.smtp_tls_var.get(),
        )

    def _send_alarm_email(self, config: AlarmConfig, value: float) -> None:
        if not config.recipient or not config.sender or not config.smtp_server:
            self._log_status("Alarm triggered but email settings are incomplete.")
            return

        subject = "HRC-110 Pressure Alarm"
        body = (
            "Pressure alarm triggered.\n"
            f"Pressure value: {value:.2f}\n"
            f"Time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = config.sender
        message["To"] = config.recipient
        message.set_content(body)

        context = ssl.create_default_context()
        try:
            if config.use_tls:
                with smtplib.SMTP(config.smtp_server, config.smtp_port) as server:
                    server.starttls(context=context)
                    if config.username:
                        server.login(config.username, config.password)
                    server.send_message(message)
            else:
                with smtplib.SMTP_SSL(
                    config.smtp_server, config.smtp_port, context=context
                ) as server:
                    if config.username:
                        server.login(config.username, config.password)
                    server.send_message(message)
            self._log_status("Alarm email sent.")
        except Exception as exc:
            self._log_status(f"Alarm email failed: {exc}")

    def _update_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _update_gemini_cost_label(self) -> None:
        if self._selected_ocr_method() == OCR_METHOD_GEMINI:
            text = f"Gemini total cost (since start): ${self.gemini_total_cost_usd:.6f}"
        else:
            text = "Gemini total cost (since start): n/a (Local OCR selected)"
        self.gemini_cost_label.configure(text=text)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def _append_status_log(self, timestamp: dt.datetime, text: str) -> None:
        path = self.status_log_var.get().strip() or self.status_log_path
        self._ensure_parent_dir(path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{timestamp.isoformat()} {text}\n")

    def _log_status(self, text: str) -> None:
        now = dt.datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        self.status_text.configure(state="normal")
        self.status_text.insert("end", f"[{timestamp}] {text}\n")
        self.status_text.configure(state="disabled")
        self.status_text.see("end")
        self._append_status_log(now, text)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = PressureMonitorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop_monitoring(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
