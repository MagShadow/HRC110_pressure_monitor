#!/usr/bin/env python3
"""GUI monitor for HRC-100/110 pressure OCR readings."""

from __future__ import annotations

import csv
import datetime as dt
import os
import smtplib
import ssl
import tempfile
import tkinter as tk
from dataclasses import dataclass
from email.message import EmailMessage
from tkinter import messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pressure_ocr import read_pressure_temperature

MIN_INTERVAL_MINUTES = 2
DATA_DIR = "data"
CSV_NAME = "pressure_readings.csv"


@dataclass
class AlarmConfig:
    enabled: bool
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

        self.monitoring = False
        self.after_id: str | None = None
        self.timestamps: list[dt.datetime] = []
        self.values: list[float] = []
        self.last_alarm_active = False
        self.last_alarm_sent: dt.datetime | None = None

        self._build_layout()
        self._refresh_cameras()

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

        ttk.Button(
            camera_frame, text="Test Camera + OCR", command=self._test_camera
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        self.camera_status = ttk.Label(camera_frame, text="")
        self.camera_status.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_alarm_panel(self, parent: ttk.Frame) -> None:
        alarm_frame = ttk.LabelFrame(parent, text="Alarm & Email", padding=10)
        alarm_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        parent.rowconfigure(1, weight=1)

        ttk.Label(alarm_frame, text="Interval (minutes):").grid(
            row=0, column=0, sticky="w"
        )
        self.interval_var = tk.StringVar(value="2")
        ttk.Entry(alarm_frame, textvariable=self.interval_var, width=10).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(alarm_frame, text=f"(min {MIN_INTERVAL_MINUTES})").grid(
            row=0, column=2, sticky="w"
        )

        self.alarm_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            alarm_frame, text="Enable alarm", variable=self.alarm_enabled_var
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(alarm_frame, text="Low threshold:").grid(row=2, column=0, sticky="w")
        self.low_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.low_var, width=10).grid(
            row=2, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="High threshold:").grid(row=3, column=0, sticky="w")
        self.high_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.high_var, width=10).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="Email recipient:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self.recipient_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.recipient_var, width=24).grid(
            row=4, column=1, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Label(alarm_frame, text="Email sender:").grid(row=5, column=0, sticky="w")
        self.sender_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.sender_var, width=24).grid(
            row=5, column=1, columnspan=2, sticky="w"
        )

        ttk.Label(alarm_frame, text="SMTP server:").grid(row=6, column=0, sticky="w")
        self.smtp_server_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_server_var, width=24).grid(
            row=6, column=1, columnspan=2, sticky="w"
        )

        ttk.Label(alarm_frame, text="SMTP port:").grid(row=7, column=0, sticky="w")
        self.smtp_port_var = tk.StringVar(value="587")
        ttk.Entry(alarm_frame, textvariable=self.smtp_port_var, width=10).grid(
            row=7, column=1, sticky="w"
        )

        ttk.Label(alarm_frame, text="SMTP user:").grid(row=8, column=0, sticky="w")
        self.smtp_user_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_user_var, width=24).grid(
            row=8, column=1, columnspan=2, sticky="w"
        )

        ttk.Label(alarm_frame, text="SMTP password:").grid(row=9, column=0, sticky="w")
        self.smtp_pass_var = tk.StringVar()
        ttk.Entry(alarm_frame, textvariable=self.smtp_pass_var, width=24, show="*").grid(
            row=9, column=1, columnspan=2, sticky="w"
        )

        self.smtp_tls_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            alarm_frame, text="Use TLS", variable=self.smtp_tls_var
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(4, 0))

        button_row = ttk.Frame(alarm_frame)
        button_row.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(10, 0))
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
        reading_frame.rowconfigure(2, weight=1)

        self.current_value_label = ttk.Label(
            reading_frame, text="--", font=("Arial", 28, "bold")
        )
        self.current_value_label.grid(row=0, column=0, sticky="w")

        self.status_label = ttk.Label(reading_frame, text="Idle")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(6, 4))

        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_xlabel("Time")
        self.axis.set_ylabel("Pressure")
        self.axis.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=reading_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", pady=(4, 8))

        status_frame = ttk.LabelFrame(reading_frame, text="Status Log")
        status_frame.grid(row=3, column=0, sticky="nsew")
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.status_text = tk.Text(status_frame, height=6, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

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

    def _test_camera(self) -> None:
        self._log_status("Testing camera...")
        self._update_status("Testing camera")
        try:
            reading = self._capture_and_ocr(keep_photo=True)
        except Exception as exc:
            messagebox.showerror(
                "Camera/OCR error",
                f"{exc}\n\nLast capture saved to data/last_capture.jpg (if available).",
            )
            self._update_status("Camera test failed")
            return
        if reading is not None:
            self._update_status(f"Test OK: {reading:.2f}")
            messagebox.showinfo("Test result", f"Pressure reading: {reading:.2f}")

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

        delay_ms = 1000 if initial else interval * 60 * 1000
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(delay_ms, self._run_cycle)

    def _run_cycle(self) -> None:
        if not self.monitoring:
            return
        self._update_status("Capturing photo")
        try:
            value = self._capture_and_ocr()
        except Exception as exc:
            self._update_status("OCR error")
            messagebox.showerror("OCR error", str(exc))
            self._log_status(f"OCR error: {exc}")
            self._schedule_next_run()
            return

        if value is not None:
            self._record_value(value)
            self._check_alarm(value)
        self._schedule_next_run()

    def _capture_and_ocr(self, keep_photo: bool = False) -> float | None:
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
        keep_path = None
        if keep_photo:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            keep_path = os.path.join(DATA_DIR, "last_capture.jpg")
            cv2.imwrite(keep_path, frame)
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)

        image_path = keep_path or temp_path
        if not image_path:
            raise RuntimeError("Unable to create capture image")

        self._update_status("Running OCR")
        try:
            pressure, _temperature = read_pressure_temperature(image_path)
            value = float(pressure.value)
            self._log_status(
                f"OCR result: {pressure.value} (confidence={pressure.confidence:.2f})"
            )
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        if keep_path:
            self._log_status(f"Saved capture to {keep_path}")

        return value

    def _record_value(self, value: float) -> None:
        now = dt.datetime.now()
        self.timestamps.append(now)
        self.values.append(value)
        self.current_value_label.configure(text=f"{value:.2f}")
        self._update_plot()
        self._append_csv(now, value)
        self._update_status("Reading captured")

    def _append_csv(self, timestamp: dt.datetime, value: float) -> None:
        path = os.path.join(DATA_DIR, CSV_NAME)
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
            times = [ts.strftime("%H:%M") for ts in self.timestamps]
            self.axis.plot(range(len(times)), self.values, marker="o", linestyle="-")
            self.axis.set_xlabel("Time")
            self.axis.set_ylabel("Pressure")
            self.axis.set_xticks(range(len(times)))
            self.axis.set_xticklabels(times, rotation=45, ha="right")
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
            self._send_alarm_email(config, value)
            messagebox.showwarning(
                "Pressure alarm", f"Pressure value out of range: {value:.2f}"
            )
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
        return AlarmConfig(
            enabled=self.alarm_enabled_var.get(),
            low_threshold=low,
            high_threshold=high,
            recipient=self.recipient_var.get().strip(),
            sender=self.sender_var.get().strip(),
            smtp_server=self.smtp_server_var.get().strip(),
            smtp_port=int(self.smtp_port_var.get() or 0),
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

    def _log_status(self, text: str) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        self.status_text.configure(state="normal")
        self.status_text.insert("end", f"[{timestamp}] {text}\n")
        self.status_text.configure(state="disabled")
        self.status_text.see("end")


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
