#!/usr/bin/env python3
"""PySide6 GUI monitor for HRC-100/110 pressure OCR readings."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import smtplib
import ssl
import tempfile
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import cv2
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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

try:
    import keyring  # type: ignore
except Exception:  # pragma: no cover - optional secure storage
    keyring = None


MIN_INTERVAL_MINUTES = 1
DEFAULT_INTERVAL_MINUTES = 10
DATA_DIR = Path("data")
DEFAULT_LOG_DIR = DATA_DIR / "logs"
DEFAULT_CONFIG_PATH = DATA_DIR / "monitor_config.json"
OCR_METHOD_LOCAL = "Local"
OCR_METHOD_GEMINI = "GEMINI"
OCR_METHODS = (OCR_METHOD_LOCAL, OCR_METHOD_GEMINI)
KEYRING_SERVICE = "HRC110_pressure_monitor"
KEYRING_USERNAME = "gemini_api_key"


class DateAxisItem(pg.DateAxisItem):
    """Date axis with explicit absolute timestamp labels."""

    def tickStrings(self, values, scale, spacing):  # type: ignore[override]
        labels: list[str] = []
        for value in values:
            labels.append(dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S"))
        return labels


@dataclass
class AlarmConfig:
    enabled: bool
    low_threshold: float | None
    high_threshold: float | None
    beep_enabled: bool
    email_enabled: bool
    recipient_text: str
    sender: str
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool


class PressureMonitorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HRC-110 Pressure Monitor")
        self.resize(1260, 800)

        self.monitoring = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._run_cycle)

        self.time_data: list[float] = []
        self.value_data: list[float] = []
        self.last_alarm_active = False
        self.gemini_total_cost_usd = 0.0

        self.active_reading_log_path: Path | None = None
        self.active_status_log_path: Path | None = None
        self.config_path = DEFAULT_CONFIG_PATH

        self._build_ui()
        self._build_email_dialog()
        self._refresh_cameras()
        self._load_config_from_path(self.config_path, silent=True)
        self._load_gemini_api_key_from_keyring()
        self._update_gemini_cost_label()

    def _build_ui(self) -> None:
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

        self.camera_group = QGroupBox("Camera")
        camera_layout = QGridLayout(self.camera_group)

        self.camera_combo = QComboBox()
        self.refresh_camera_btn = QPushButton("Refresh")
        self.refresh_camera_btn.clicked.connect(self._refresh_cameras)

        self.ocr_method_combo = QComboBox()
        self.ocr_method_combo.addItems(list(OCR_METHODS))
        self.ocr_method_combo.currentTextChanged.connect(lambda _text: self._update_gemini_cost_label())

        self.gemini_api_key_edit = QLineEdit()
        self.gemini_api_key_edit.setEchoMode(QLineEdit.Password)

        self.remember_key_check = QCheckBox("Remember API key (encrypted)")
        if keyring is None:
            self.remember_key_check.setEnabled(False)
            self.remember_key_check.setToolTip("keyring is unavailable in this environment")
        self.remember_key_check.toggled.connect(self._on_remember_key_toggled)

        self.test_camera_btn = QPushButton("Test Camera")
        self.test_camera_btn.clicked.connect(self._test_camera)
        self.test_ocr_btn = QPushButton("Test OCR")
        self.test_ocr_btn.clicked.connect(self._test_ocr)

        self.camera_status = QLabel("")

        camera_layout.addWidget(QLabel("Select camera index:"), 0, 0)
        camera_layout.addWidget(self.camera_combo, 0, 1)
        camera_layout.addWidget(self.refresh_camera_btn, 0, 2)
        camera_layout.addWidget(QLabel("OCR method:"), 1, 0)
        camera_layout.addWidget(self.ocr_method_combo, 1, 1)
        camera_layout.addWidget(QLabel("Gemini API key:"), 2, 0)
        camera_layout.addWidget(self.gemini_api_key_edit, 2, 1, 1, 2)
        camera_layout.addWidget(self.remember_key_check, 3, 0, 1, 3)
        camera_layout.addWidget(self.test_camera_btn, 4, 0, 1, 2)
        camera_layout.addWidget(self.test_ocr_btn, 4, 2)
        camera_layout.addWidget(self.camera_status, 5, 0, 1, 3)

        self.alarm_group = QGroupBox("Alarm & Logging")
        alarm_layout = QGridLayout(self.alarm_group)

        self.log_folder_edit = QLineEdit(str(DEFAULT_LOG_DIR))
        self.log_folder_browse_btn = QPushButton("Browse")
        self.log_folder_browse_btn.clicked.connect(self._browse_log_folder)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(MIN_INTERVAL_MINUTES, 24 * 60)
        self.interval_spin.setValue(DEFAULT_INTERVAL_MINUTES)
        self.interval_spin.setSuffix(" min")

        self.alarm_enabled_check = QCheckBox("Enable alarm")
        self.low_enabled_check = QCheckBox("Low threshold")
        self.low_threshold_spin = pg.SpinBox(value=0.0, bounds=(-1000.0, 10000.0), step=0.01)
        self.high_enabled_check = QCheckBox("High threshold")
        self.high_threshold_spin = pg.SpinBox(value=0.0, bounds=(-1000.0, 10000.0), step=0.01)

        self.beep_alarm_check = QCheckBox("Beep alarm")
        self.beep_alarm_check.setChecked(True)
        self.email_alarm_check = QCheckBox("Email alarm")
        self.email_settings_btn = QPushButton("Email Settings...")
        self.email_settings_btn.clicked.connect(self._open_email_settings_dialog)

        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self._load_config)
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start_monitoring)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_monitoring)

        alarm_layout.addWidget(QLabel("Log folder:"), 0, 0)
        alarm_layout.addWidget(self.log_folder_edit, 0, 1)
        alarm_layout.addWidget(self.log_folder_browse_btn, 0, 2)
        alarm_layout.addWidget(QLabel("Interval:"), 1, 0)
        alarm_layout.addWidget(self.interval_spin, 1, 1)
        alarm_layout.addWidget(self.alarm_enabled_check, 2, 0, 1, 3)
        alarm_layout.addWidget(self.low_enabled_check, 3, 0)
        alarm_layout.addWidget(self.low_threshold_spin, 3, 1)
        alarm_layout.addWidget(self.high_enabled_check, 4, 0)
        alarm_layout.addWidget(self.high_threshold_spin, 4, 1)
        alarm_layout.addWidget(self.beep_alarm_check, 5, 0)
        alarm_layout.addWidget(self.email_alarm_check, 5, 1)
        alarm_layout.addWidget(self.email_settings_btn, 5, 2)
        alarm_layout.addWidget(self.load_config_btn, 6, 0)
        alarm_layout.addWidget(self.save_config_btn, 6, 1)
        alarm_layout.addWidget(self.start_btn, 7, 0)
        alarm_layout.addWidget(self.stop_btn, 7, 1)

        self.readout_group = QGroupBox("Pressure Reading")
        readout_layout = QVBoxLayout(self.readout_group)

        self.current_value_label = QLabel("--")
        self.current_value_label.setStyleSheet("color: #b42318; font-size: 36px; font-weight: 700;")
        self.status_label = QLabel("Idle")
        self.gemini_cost_label = QLabel("Gemini total cost (since start): n/a")

        self.plot_widget = pg.PlotWidget(axisItems={"bottom": DateAxisItem()})
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Time", color="k")
        self.plot_widget.setLabel("left", "Pressure", color="k")
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen(color="k"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen(color="k"))
        self.plot_curve = self.plot_widget.plot([], [], pen=pg.mkPen(color=(15, 70, 200), width=2), symbol="o")

        self.status_log_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout(self.status_log_group)
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        readout_layout.addWidget(self.current_value_label)
        readout_layout.addWidget(self.status_label)
        readout_layout.addWidget(self.gemini_cost_label)
        readout_layout.addWidget(self.plot_widget, 1)
        readout_layout.addWidget(self.status_log_group)

        left_layout.addWidget(self.camera_group)
        left_layout.addWidget(self.alarm_group)
        left_layout.addStretch(1)
        right_layout.addWidget(self.readout_group)

        self.setCentralWidget(central)

    def _build_email_dialog(self) -> None:
        self.email_dialog = QDialog(self)
        self.email_dialog.setWindowTitle("Email Settings")
        self.email_dialog.setModal(True)

        layout = QVBoxLayout(self.email_dialog)
        form = QFormLayout()

        self.email_recipient_edit = QLineEdit()
        self.email_recipient_edit.setPlaceholderText("user1@example.com; user2@example.com")
        self.email_sender_edit = QLineEdit()
        self.smtp_server_edit = QLineEdit()
        self.smtp_port_spin = QSpinBox()
        self.smtp_port_spin.setRange(1, 65535)
        self.smtp_port_spin.setValue(587)
        self.smtp_user_edit = QLineEdit()
        self.smtp_password_edit = QLineEdit()
        self.smtp_password_edit.setEchoMode(QLineEdit.Password)
        self.smtp_tls_check = QCheckBox("Use TLS")
        self.smtp_tls_check.setChecked(True)

        form.addRow("Recipient(s)", self.email_recipient_edit)
        form.addRow("Sender", self.email_sender_edit)
        form.addRow("SMTP server", self.smtp_server_edit)
        form.addRow("SMTP port", self.smtp_port_spin)
        form.addRow("SMTP user", self.smtp_user_edit)
        form.addRow("SMTP password", self.smtp_password_edit)
        form.addRow("", self.smtp_tls_check)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.email_dialog.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _open_email_settings_dialog(self) -> None:
        self.email_dialog.show()
        self.email_dialog.raise_()
        self.email_dialog.activateWindow()

    def _on_remember_key_toggled(self, checked: bool) -> None:
        if checked:
            self._load_gemini_api_key_from_keyring()

    def _refresh_cameras(self) -> None:
        available: list[str] = []
        for index in range(6):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                available.append(str(index))
            capture.release()

        if not available:
            available = ["0"]
            self.camera_status.setText("No camera detected. Defaulting to 0.")
        else:
            self.camera_status.setText(f"Detected cameras: {', '.join(available)}")

        current = self.camera_combo.currentText()
        self.camera_combo.clear()
        self.camera_combo.addItems(available)
        if current in available:
            self.camera_combo.setCurrentText(current)

    def _browse_log_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Log Folder",
            self.log_folder_edit.text().strip() or str(DEFAULT_LOG_DIR),
        )
        if folder:
            self.log_folder_edit.setText(folder)

    def _test_camera(self) -> None:
        self._update_status("Testing camera")
        self._log_status("Testing camera capture...")
        try:
            image_path, temp_path = self._capture_photo(keep_photo=True)
        except Exception as exc:
            QMessageBox.critical(self, "Camera error", str(exc))
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
        QMessageBox.information(self, "Camera test", f"Capture successful.\nSaved to {image_path}.")

    def _test_ocr(self) -> None:
        method = self._selected_ocr_method()
        self._update_status("Testing OCR")
        self._log_status(f"Testing OCR ({method})...")
        try:
            value, request_cost = self._capture_and_ocr(keep_photo=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "OCR test error",
                f"{exc}\n\nLast capture saved to data/last_capture.jpg (if available).",
            )
            self._update_status("OCR test failed")
            return

        self._update_status(f"OCR test OK: {value:.2f}")
        if method == OCR_METHOD_GEMINI and request_cost is not None:
            QMessageBox.information(
                self,
                "OCR test",
                f"Pressure reading: {value:.2f}\nEstimated request cost: ${request_cost:.6f}",
            )
        else:
            QMessageBox.information(self, "OCR test", f"Pressure reading: {value:.2f}")

    def _start_monitoring(self) -> None:
        if self.monitoring:
            return

        if self.interval_spin.value() < MIN_INTERVAL_MINUTES:
            QMessageBox.critical(
                self,
                "Invalid interval",
                f"Minimum interval is {MIN_INTERVAL_MINUTES} minutes.",
            )
            return

        self._persist_gemini_api_key_to_keyring()
        self._start_new_log_files()

        self.time_data = []
        self.value_data = []
        self.plot_curve.setData([], [])
        self.last_alarm_active = False
        self.gemini_total_cost_usd = 0.0
        self._update_gemini_cost_label()

        self.monitoring = True
        self._update_status("Monitoring started")
        self._log_status("Monitoring started")

        # Immediate first run, then periodic runs.
        self._run_cycle()
        self.timer.start(self.interval_spin.value() * 60 * 1000)

    def _stop_monitoring(self) -> None:
        if not self.monitoring:
            return

        self.timer.stop()
        self.monitoring = False
        self._update_status("Monitoring stopped")
        self._log_status("Monitoring stopped")
        self.active_reading_log_path = None
        self.active_status_log_path = None

    def _start_new_log_files(self) -> None:
        folder = Path(self.log_folder_edit.text().strip() or str(DEFAULT_LOG_DIR))
        folder.mkdir(parents=True, exist_ok=True)
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.active_reading_log_path = folder / f"pressure_log_{timestamp}.csv"
        self.active_status_log_path = folder / f"pressure_status_{timestamp}.log"

        with self.active_reading_log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "pressure"])

        with self.active_status_log_path.open("w", encoding="utf-8"):
            pass

    def _run_cycle(self) -> None:
        if not self.monitoring:
            return

        self._update_status("Capturing photo")
        try:
            value, request_cost = self._capture_and_ocr(keep_photo=False)
        except Exception as exc:
            self._update_status("OCR error")
            self._log_status(f"OCR error: {exc}")
            QMessageBox.critical(self, "OCR error", str(exc))
            return

        self._record_value(value)

        if self._selected_ocr_method() == OCR_METHOD_GEMINI and request_cost is not None:
            self.gemini_total_cost_usd += request_cost
            self._update_gemini_cost_label()

        self._check_alarm(value)

    def _selected_ocr_method(self) -> str:
        method = self.ocr_method_combo.currentText().strip()
        if method not in OCR_METHODS:
            return OCR_METHOD_LOCAL
        return method

    def _capture_photo(self, keep_photo: bool) -> tuple[str, str | None]:
        camera_index = int(self.camera_combo.currentText() or "0")
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

        success, frame = capture.read()
        capture.release()
        if not success:
            raise RuntimeError("Unable to capture photo from camera")

        now = dt.datetime.now()
        self._log_status(f"Photo captured at {now:%Y-%m-%d %H:%M:%S}")

        temp_path = None
        if keep_photo:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            image_path = DATA_DIR / "last_capture.jpg"
            cv2.imwrite(str(image_path), frame)
            return str(image_path), None

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
        cv2.imwrite(temp_path, frame)
        return temp_path, temp_path

    def _capture_and_ocr(self, keep_photo: bool) -> tuple[float, float | None]:
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

                api_key = self.gemini_api_key_edit.text().strip() or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "Gemini API key is missing. Enter it in the GUI field, "
                        "enable secure storage, or set GEMINI_API_KEY."
                    )

                result = read_gemini_ocr(
                    image_path=Path(image_path),
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
        epoch_ts = now.timestamp()
        self.time_data.append(epoch_ts)
        self.value_data.append(value)

        self.plot_curve.setData(self.time_data, self.value_data)
        self.current_value_label.setText(f"{value:.2f}")
        self._append_reading_log(now, value)
        self._update_status("Reading captured")

    def _append_reading_log(self, timestamp: dt.datetime, value: float) -> None:
        if self.active_reading_log_path is None:
            return
        with self.active_reading_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([timestamp.isoformat(), f"{value:.2f}"])

    def _check_alarm(self, value: float) -> None:
        config = self._alarm_config()
        if not config.enabled:
            self.last_alarm_active = False
            return

        outside_low = config.low_threshold is not None and value < config.low_threshold
        outside_high = config.high_threshold is not None and value > config.high_threshold
        alarm_active = outside_low or outside_high

        if alarm_active and not self.last_alarm_active:
            if config.beep_enabled:
                QApplication.beep()
                self._log_status(f"Beep alarm triggered at pressure={value:.2f}")
            if config.email_enabled:
                self._send_alarm_email(config, value)
            if not config.beep_enabled and not config.email_enabled:
                self._log_status("Alarm triggered but both beep/email alarms are disabled.")

        self.last_alarm_active = alarm_active

    def _alarm_config(self) -> AlarmConfig:
        low = float(self.low_threshold_spin.value()) if self.low_enabled_check.isChecked() else None
        high = float(self.high_threshold_spin.value()) if self.high_enabled_check.isChecked() else None

        return AlarmConfig(
            enabled=self.alarm_enabled_check.isChecked(),
            low_threshold=low,
            high_threshold=high,
            beep_enabled=self.beep_alarm_check.isChecked(),
            email_enabled=self.email_alarm_check.isChecked(),
            recipient_text=self.email_recipient_edit.text().strip(),
            sender=self.email_sender_edit.text().strip(),
            smtp_server=self.smtp_server_edit.text().strip(),
            smtp_port=int(self.smtp_port_spin.value()),
            username=self.smtp_user_edit.text().strip(),
            password=self.smtp_password_edit.text(),
            use_tls=self.smtp_tls_check.isChecked(),
        )

    @staticmethod
    def _parse_recipients(text: str) -> list[str]:
        parts = re.split(r"[;,\n]+", text)
        return [part.strip() for part in parts if part.strip()]

    def _send_alarm_email(self, config: AlarmConfig, value: float) -> None:
        recipients = self._parse_recipients(config.recipient_text)
        if not recipients or not config.sender or not config.smtp_server:
            self._log_status("Alarm triggered but email settings are incomplete.")
            return

        body = (
            "Pressure alarm triggered.\n"
            f"Pressure value: {value:.2f}\n"
            f"Time: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )

        message = EmailMessage()
        message["Subject"] = "HRC-110 Pressure Alarm"
        message["From"] = config.sender
        message["To"] = ", ".join(recipients)
        message.set_content(body)

        context = ssl.create_default_context()
        try:
            if config.use_tls:
                with smtplib.SMTP(config.smtp_server, config.smtp_port, timeout=20) as server:
                    server.starttls(context=context)
                    if config.username:
                        server.login(config.username, config.password)
                    server.send_message(message)
            else:
                with smtplib.SMTP_SSL(config.smtp_server, config.smtp_port, context=context, timeout=20) as server:
                    if config.username:
                        server.login(config.username, config.password)
                    server.send_message(message)
            self._log_status("Alarm email sent.")
        except Exception as exc:
            self._log_status(f"Alarm email failed: {exc}")

    def _update_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _update_gemini_cost_label(self) -> None:
        if self._selected_ocr_method() == OCR_METHOD_GEMINI:
            self.gemini_cost_label.setText(
                f"Gemini total cost (since start): ${self.gemini_total_cost_usd:.6f}"
            )
        else:
            self.gemini_cost_label.setText("Gemini total cost (since start): n/a (Local OCR selected)")

    def _append_status_log_file(self, timestamp: dt.datetime, text: str) -> None:
        if self.active_status_log_path is None:
            return
        with self.active_status_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp.isoformat()} {text}\n")

    def _log_status(self, text: str) -> None:
        now = dt.datetime.now()
        line = f"[{now:%H:%M:%S}] {text}"
        self.status_text.appendPlainText(line)
        self._append_status_log_file(now, text)

    def _build_config_payload(self) -> dict[str, object]:
        return {
            "camera_index": self.camera_combo.currentText(),
            "ocr_method": self._selected_ocr_method(),
            "remember_gemini_api_key": self.remember_key_check.isChecked(),
            "log_folder": self.log_folder_edit.text().strip(),
            "interval_minutes": int(self.interval_spin.value()),
            "alarm_enabled": self.alarm_enabled_check.isChecked(),
            "low_enabled": self.low_enabled_check.isChecked(),
            "low_threshold": float(self.low_threshold_spin.value()),
            "high_enabled": self.high_enabled_check.isChecked(),
            "high_threshold": float(self.high_threshold_spin.value()),
            "beep_alarm_enabled": self.beep_alarm_check.isChecked(),
            "email_alarm_enabled": self.email_alarm_check.isChecked(),
            "email_recipients": self.email_recipient_edit.text().strip(),
            "email_sender": self.email_sender_edit.text().strip(),
            "smtp_server": self.smtp_server_edit.text().strip(),
            "smtp_port": int(self.smtp_port_spin.value()),
            "smtp_user": self.smtp_user_edit.text().strip(),
            "smtp_password": self.smtp_password_edit.text(),
            "smtp_use_tls": self.smtp_tls_check.isChecked(),
        }

    def _apply_config_payload(self, payload: dict[str, object]) -> None:
        camera_index = str(payload.get("camera_index", "")).strip()
        if camera_index:
            idx = self.camera_combo.findText(camera_index)
            if idx >= 0:
                self.camera_combo.setCurrentIndex(idx)

        ocr_method = str(payload.get("ocr_method", OCR_METHOD_LOCAL))
        idx = self.ocr_method_combo.findText(ocr_method)
        if idx >= 0:
            self.ocr_method_combo.setCurrentIndex(idx)

        remember = bool(payload.get("remember_gemini_api_key", False))
        self.remember_key_check.setChecked(remember)
        if remember:
            self._load_gemini_api_key_from_keyring()
        else:
            self.gemini_api_key_edit.setText("")

        self.log_folder_edit.setText(str(payload.get("log_folder", str(DEFAULT_LOG_DIR))))
        self.interval_spin.setValue(max(MIN_INTERVAL_MINUTES, int(payload.get("interval_minutes", DEFAULT_INTERVAL_MINUTES))))

        self.alarm_enabled_check.setChecked(bool(payload.get("alarm_enabled", False)))
        self.low_enabled_check.setChecked(bool(payload.get("low_enabled", False)))
        self.low_threshold_spin.setValue(float(payload.get("low_threshold", 0.0)))
        self.high_enabled_check.setChecked(bool(payload.get("high_enabled", False)))
        self.high_threshold_spin.setValue(float(payload.get("high_threshold", 0.0)))
        self.beep_alarm_check.setChecked(bool(payload.get("beep_alarm_enabled", True)))
        self.email_alarm_check.setChecked(bool(payload.get("email_alarm_enabled", False)))

        self.email_recipient_edit.setText(str(payload.get("email_recipients", "")))
        self.email_sender_edit.setText(str(payload.get("email_sender", "")))
        self.smtp_server_edit.setText(str(payload.get("smtp_server", "")))
        self.smtp_port_spin.setValue(int(payload.get("smtp_port", 587)))
        self.smtp_user_edit.setText(str(payload.get("smtp_user", "")))
        self.smtp_password_edit.setText(str(payload.get("smtp_password", "")))
        self.smtp_tls_check.setChecked(bool(payload.get("smtp_use_tls", True)))

        self._update_gemini_cost_label()

    def _save_config(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config",
            str(self.config_path),
            "JSON files (*.json);;All files (*.*)",
        )
        if not selected:
            return

        payload = self._build_config_payload()
        path = Path(selected)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.config_path = path
        self._persist_gemini_api_key_to_keyring()
        self._log_status(f"Config saved to {path}")

    def _load_config(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config",
            str(self.config_path.parent),
            "JSON files (*.json);;All files (*.*)",
        )
        if not selected:
            return
        self._load_config_from_path(Path(selected), silent=False)

    def _load_config_from_path(self, path: Path, silent: bool) -> None:
        if not path.exists():
            if not silent:
                QMessageBox.critical(self, "Config error", f"Config file not found: {path}")
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Config root must be a JSON object")
            self._apply_config_payload(payload)
            self.config_path = path
            if not silent:
                self._log_status(f"Config loaded from {path}")
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Config error", f"Failed to load config:\n{exc}")

    def _load_gemini_api_key_from_keyring(self) -> None:
        if keyring is None or not self.remember_key_check.isChecked():
            return
        try:
            stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            stored = None
        if stored:
            self.gemini_api_key_edit.setText(stored)

    def _persist_gemini_api_key_to_keyring(self) -> None:
        if keyring is None:
            if self.remember_key_check.isChecked():
                self._log_status("keyring unavailable; Gemini API key not persisted securely")
            return

        api_key = self.gemini_api_key_edit.text().strip()
        if self.remember_key_check.isChecked() and api_key:
            try:
                keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, api_key)
            except Exception:
                self._log_status("Failed to save Gemini API key to keyring")
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
            except Exception:
                pass

    def _save_default_config_on_exit(self) -> None:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(
                json.dumps(self._build_config_payload(), indent=2), encoding="utf-8"
            )
            self._persist_gemini_api_key_to_keyring()
        except Exception:
            # Exit path should not block app close.
            pass

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        self._stop_monitoring()
        self._save_default_config_on_exit()
        event.accept()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    pg.setConfigOptions(antialias=True)
    window = PressureMonitorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
