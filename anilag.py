#!/usr/bin/env python3
"""
Anilag - Rice Weevil Detection System
Single-file version for easy deployment on Raspberry Pi 5
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QTextEdit, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from typing import Optional, List, Dict, Tuple
from ultralytics import YOLO
import threading
import csv
import os
from datetime import datetime
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv


# ============================================================================
# DETECTION MODULE
# ============================================================================

class DetectionResult:
    def __init__(self, boxes: List, confidences: List[float], class_ids: List[int], class_names: List[str]):
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids
        self.class_names = class_names
        self.count = len(boxes)

    def has_detections(self) -> bool:
        return self.count > 0


class YOLOv11Detector:
    def __init__(self, model_path: str = 'yolov11n.pt', confidence_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model: Optional[YOLO] = None
        self.class_names = []
        self.lock = threading.Lock()
        self.initialized = False

    def initialize(self) -> bool:
        try:
            with self.lock:
                self.model = YOLO(self.model_path)
                self.class_names = self.model.names
                self.initialized = True
                print(f"YOLOv11 model loaded from {self.model_path}")
                print(f"Classes: {self.class_names}")
                return True
        except Exception as e:
            print(f"YOLO model initialization error: {e}")
            return False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.initialized or self.model is None:
            return DetectionResult([], [], [], [])

        try:
            with self.lock:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            
            boxes = []
            confidences = []
            class_ids = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                        confidences.append(float(box.conf[0].cpu().numpy()))
                        class_ids.append(int(box.cls[0].cpu().numpy()))
            
            return DetectionResult(boxes, confidences, class_ids, self.class_names)
        except Exception as e:
            print(f"Detection error: {e}")
            return DetectionResult([], [], [], [])

    def draw_detections(self, frame: np.ndarray, detection: DetectionResult, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for i, (box, conf, cls_id) in enumerate(zip(detection.boxes, detection.confidences, detection.class_ids)):
            x1, y1, x2, y2 = box
            class_name = detection.class_names[cls_id] if cls_id < len(detection.class_names) else f"Class {cls_id}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        count_text = f"Count: {detection.count}"
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame


# ============================================================================
# HARDWARE MODULES
# ============================================================================

class Camera:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_callback: Optional = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.current_frame = None

    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False

    def start(self, frame_callback: Optional = None) -> bool:
        if self.running:
            return True
        
        if not self.initialize():
            return False
        
        self.frame_callback = frame_callback
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.current_frame = frame.copy()
                    if self.frame_callback:
                        self.frame_callback(frame)
                else:
                    print("Failed to read frame from camera")
            except Exception as e:
                print(f"Capture loop error: {e}")
                break

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.cap = None

    def is_running(self) -> bool:
        return self.running

    def __del__(self):
        self.stop()


class TemperatureSensor:
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.sensor_path = None
        self.initialized = False

    def initialize(self) -> bool:
        try:
            base_dir = '/sys/bus/w1/devices/'
            if self.device_id:
                self.sensor_path = os.path.join(base_dir, self.device_id, 'w1_slave')
                if os.path.exists(self.sensor_path):
                    self.initialized = True
                    return True
                else:
                    print(f"Sensor device {self.device_id} not found")
                    return False
            else:
                if not os.path.exists(base_dir):
                    print("1-Wire devices directory not found. Enable 1-Wire interface in raspi-config")
                    return False
                
                for device in os.listdir(base_dir):
                    if device.startswith('28-'):
                        self.device_id = device
                        self.sensor_path = os.path.join(base_dir, device, 'w1_slave')
                        self.initialized = True
                        print(f"Auto-detected temperature sensor: {device}")
                        return True
                
                print("No DS18B20 sensor found")
                return False
        except Exception as e:
            print(f"Temperature sensor initialization error: {e}")
            return False

    def read_temperature(self) -> Optional[float]:
        if not self.initialized or not self.sensor_path:
            return None
        
        try:
            with open(self.sensor_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            if 'YES' not in lines[0]:
                return None
            
            temp_str = lines[1].split('t=')[-1].strip()
            temp_celsius = float(temp_str) / 1000.0
            return temp_celsius
        except Exception as e:
            print(f"Temperature read error: {e}")
            return None


class LEDController:
    def __init__(self, gpio_pin: int = 18, led_count: int = 60, brightness: int = 255):
        self.gpio_pin = gpio_pin
        self.led_count = led_count
        self.brightness = brightness
        self.strip = None
        self.initialized = False

    def initialize(self) -> bool:
        try:
            from rpi_ws281x import PixelStrip, ws
            
            LED_STRIP = ws.WS2813_STRIP
            LED_CHANNEL = 0
            
            self.strip = PixelStrip(
                self.led_count,
                self.gpio_pin,
                LED_CHANNEL,
                None,
                LED_STRIP,
                800000,
                5,
                self.brightness,
                255,
                0
            )
            
            self.strip.begin()
            self.initialized = True
            return True
        except ImportError:
            print("rpi_ws281x not available. Running in simulation mode.")
            self.initialized = True
            return True
        except Exception as e:
            print(f"LED controller initialization error: {e}")
            return False

    def set_color(self, r: int, g: int, b: int):
        if not self.initialized:
            return
        
        try:
            if self.strip:
                for i in range(self.led_count):
                    self.strip.setPixelColor(i, self.strip.Color(r, g, b))
                self.strip.show()
            else:
                print(f"Simulation: Setting LEDs to RGB({r}, {g}, {b})")
        except Exception as e:
            print(f"LED color set error: {e}")

    def set_red(self):
        self.set_color(255, 0, 0)

    def set_white(self):
        self.set_color(255, 255, 255)

    def off(self):
        self.set_color(0, 0, 0)

    def cleanup(self):
        if self.strip:
            self.off()


# ============================================================================
# LOGGING MODULE
# ============================================================================

@dataclass
class DetectionLog:
    timestamp: str
    rice_weevil_count: int
    temperature_celsius: Optional[float]
    recommendation: str
    activity: str


class DetectionLogger:
    def __init__(self, log_file: str = 'logs/detection_log.csv'):
        self.log_file = log_file
        self.logs: List[DetectionLog] = []
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def initialize(self) -> bool:
        try:
            file_exists = os.path.exists(self.log_file)
            with open(self.log_file, 'a', newline='') as f:
                if not file_exists:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Rice Weevil Count', 'Temperature (°C)', 'Recommendation', 'Activity'])
            return True
        except Exception as e:
            print(f"Logger initialization error: {e}")
            return False

    def generate_recommendation(self, rice_weevil_count: int, is_after_mixing: bool = False) -> str:
        if is_after_mixing:
            if rice_weevil_count > 0:
                return "Heat Treatment Needed"
            else:
                return "Take Out Rice"
        else:
            if rice_weevil_count > 0:
                return "Mix and Sift Rice"
            else:
                return "No Action Needed"

    def log_detection(self, rice_weevil_count: int, temperature: Optional[float], 
                     is_after_mixing: bool = False, activity: str = "Detection") -> DetectionLog:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recommendation = self.generate_recommendation(rice_weevil_count, is_after_mixing)
        
        log_entry = DetectionLog(
            timestamp=timestamp,
            rice_weevil_count=rice_weevil_count,
            temperature_celsius=temperature,
            recommendation=recommendation,
            activity=activity
        )
        
        self.logs.append(log_entry)
        self._write_to_csv(log_entry)
        
        return log_entry

    def _write_to_csv(self, log_entry: DetectionLog):
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    log_entry.timestamp,
                    log_entry.rice_weevil_count,
                    log_entry.temperature_celsius,
                    log_entry.recommendation,
                    log_entry.activity
                ])
        except Exception as e:
            print(f"Error writing to CSV: {e}")


# ============================================================================
# NOTIFICATION MODULE
# ============================================================================

class EmailNotifier:
    def __init__(self, config_file: str = 'config.env'):
        load_dotenv(config_file)
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.sender_email = os.getenv('EMAIL_SENDER', '')
        self.sender_password = os.getenv('EMAIL_PASSWORD', '')
        self.recipient_email = os.getenv('EMAIL_RECIPIENT', '')
        self.enabled = bool(self.sender_email and self.sender_password and self.recipient_email)

    def initialize(self) -> bool:
        if not self.enabled:
            print("Email notification disabled. Check config.env for credentials.")
            return False
        print(f"Email notifier configured: {self.sender_email} -> {self.recipient_email}")
        return True

    def send_email(self, subject: str, body: str, is_html: bool = False) -> bool:
        if not self.enabled:
            print("Email notification disabled")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject

            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"Email sent: {subject}")
            return True
        except Exception as e:
            print(f"Email send error: {e}")
            return False

    def send_detection_alert(self, timestamp: str, rice_weevil_count: int, 
                            temperature: Optional[float], recommendation: str, 
                            activity: str) -> bool:
        subject = f"Anilag Detection Alert - {timestamp}"
        
        temp_str = f"{temperature:.2f}°C" if temperature is not None else "N/A"
        
        body = f"""
Anilag Rice Weevil Detection System
====================================

Detection Details:
- Timestamp: {timestamp}
- Activity: {activity}
- Rice Weevil Count: {rice_weevil_count}
- Temperature: {temp_str}
- Recommendation: {recommendation}

This is an automated notification from the Anilag detection system.
"""
        
        return self.send_email(subject, body)

    def send_activity_log(self, activity: str, details: str) -> bool:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"Anilag Activity Log - {timestamp}"
        body = f"""
Anilag System Activity
======================

Timestamp: {timestamp}
Activity: {activity}

Details:
{details}

This is an automated notification from the Anilag detection system.
"""
        return self.send_email(subject, body)


# ============================================================================
# GUI MODULE
# ============================================================================

class DetectionThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str)

    def __init__(self, camera: Camera, detector: YOLOv11Detector):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                detection = self.detector.detect(frame)
                self.frame_ready.emit(frame, detection)
                self.detection_update.emit(
                    detection.count,
                    0.0,
                    "Detection"
                )
            self.msleep(50)

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anilag - Rice Weevil Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        load_dotenv('config.env')
        
        self.camera = Camera(
            camera_id=int(os.getenv('CAMERA_ID', '0')),
            width=int(os.getenv('CAMERA_WIDTH', '640')),
            height=int(os.getenv('CAMERA_HEIGHT', '480')),
            fps=int(os.getenv('CAMERA_FPS', '30'))
        )
        
        self.detector = YOLOv11Detector(
            model_path=os.getenv('MODEL_PATH', 'models/yolov11n.pt'),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
            iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.45'))
        )
        
        self.temp_sensor = TemperatureSensor(
            device_id=os.getenv('TEMP_SENSOR_DEVICE_ID')
        )
        
        self.led_controller = LEDController(
            gpio_pin=int(os.getenv('LED_GPIO_PIN', '18')),
            led_count=int(os.getenv('LED_COUNT', '60')),
            brightness=int(os.getenv('LED_BRIGHTNESS', '255'))
        )
        
        self.logger = DetectionLogger(log_file=os.getenv('LOG_FILE', 'logs/detection_log.csv'))
        self.email_notifier = EmailNotifier('config.env')
        
        self.detection_thread: Optional[DetectionThread] = None
        self.is_scanning = False
        self.is_after_mixing = False
        
        self.init_ui()
        self.initialize_components()
        
        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=2)
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed - Click Start Scan")
        left_panel.addWidget(self.camera_label)
        
        detection_group = QGroupBox("Detection Information")
        detection_layout = QGridLayout()
        
        self.count_label = QLabel("Weevil Count: 0")
        self.count_label.setFont(QFont("Arial", 14, QFont.Bold))
        detection_layout.addWidget(self.count_label, 0, 0)
        
        self.temp_label = QLabel("Temperature: --°C")
        self.temp_label.setFont(QFont("Arial", 12))
        detection_layout.addWidget(self.temp_label, 0, 1)
        
        self.recommendation_label = QLabel("Recommendation: --")
        self.recommendation_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.recommendation_label.setStyleSheet("color: #0066cc;")
        detection_layout.addWidget(self.recommendation_label, 1, 0, 1, 2)
        
        detection_group.setLayout(detection_layout)
        left_panel.addWidget(detection_group)
        
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=1)
        
        scan_group = QGroupBox("Scan Controls")
        scan_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Scan")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_button.clicked.connect(self.toggle_scan)
        scan_layout.addWidget(self.start_button)
        
        self.mixing_button = QPushButton("Mark as After Mixing/Sifting")
        self.mixing_button.setMinimumHeight(40)
        self.mixing_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.mixing_button.clicked.connect(self.toggle_mixing_state)
        scan_layout.addWidget(self.mixing_button)
        
        scan_group.setLayout(scan_layout)
        right_panel.addWidget(scan_group)
        
        led_group = QGroupBox("LED Controls")
        led_layout = QHBoxLayout()
        
        self.red_light_button = QPushButton("Red Light")
        self.red_light_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.red_light_button.clicked.connect(self.set_red_light)
        led_layout.addWidget(self.red_light_button)
        
        self.white_light_button = QPushButton("White Light")
        self.white_light_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: black;
                font-size: 14px;
                border-radius: 5px;
                border: 2px solid #333;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.white_light_button.clicked.connect(self.set_white_light)
        led_layout.addWidget(self.white_light_button)
        
        self.led_off_button = QPushButton("LEDs Off")
        self.led_off_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.led_off_button.clicked.connect(self.set_leds_off)
        led_layout.addWidget(self.led_off_button)
        
        led_group.setLayout(led_layout)
        right_panel.addWidget(led_group)
        
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def initialize_components(self):
        if self.detector.initialize():
            self.log_message("YOLOv11 detector initialized successfully")
        else:
            self.log_message("Failed to initialize YOLOv11 detector")
        
        if self.temp_sensor.initialize():
            self.log_message("Temperature sensor initialized")
        else:
            self.log_message("Temperature sensor not available")
        
        if self.led_controller.initialize():
            self.log_message("LED controller initialized")
        else:
            self.log_message("LED controller not available")
        
        if self.logger.initialize():
            self.log_message("Logger initialized")
        else:
            self.log_message("Failed to initialize logger")
        
        if self.email_notifier.initialize():
            self.log_message("Email notifier initialized")
        else:
            self.log_message("Email notifier disabled")

    def toggle_scan(self):
        if self.is_scanning:
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self):
        if not self.camera.start():
            self.log_message("Failed to start camera")
            return
        
        self.detection_thread = DetectionThread(self.camera, self.detector)
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.detection_update.connect(self.update_detection)
        self.detection_thread.start()
        
        self.is_scanning = True
        self.start_button.setText("Stop Scan")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.status_label.setText("Scanning...")
        self.log_message("Scan started")
        self.email_notifier.send_activity_log("Scan Started", "Live detection has been initiated")

    def stop_scan(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        self.camera.stop()
        
        self.is_scanning = False
        self.start_button.setText("Start Scan")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.status_label.setText("Ready")
        self.log_message("Scan stopped")
        self.email_notifier.send_activity_log("Scan Stopped", "Live detection has been stopped")

    def toggle_mixing_state(self):
        self.is_after_mixing = not self.is_after_mixing
        if self.is_after_mixing:
            self.mixing_button.setText("After Mixing: ON")
            self.mixing_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 14px;
                    border-radius: 5px;
                }
            """)
            self.log_message("Marked as after mixing/sifting")
        else:
            self.mixing_button.setText("Mark as After Mixing/Sifting")
            self.mixing_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-size: 14px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            self.log_message("Marked as before mixing/sifting")

    def set_red_light(self):
        self.led_controller.set_red()
        self.log_message("Red light activated")
        self.email_notifier.send_activity_log("LED Control", "Red light activated to lure rice weevils")

    def set_white_light(self):
        self.led_controller.set_white()
        self.log_message("White light activated")
        self.email_notifier.send_activity_log("LED Control", "White light activated for detection")

    def set_leds_off(self):
        self.led_controller.off()
        self.log_message("LEDs turned off")
        self.email_notifier.send_activity_log("LED Control", "LEDs turned off")

    def update_frame(self, frame: np.ndarray, detection: DetectionResult):
        annotated_frame = self.detector.draw_detections(frame, detection)
        
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)

    def update_detection(self, count: int, confidence: float, activity: str):
        self.count_label.setText(f"Weevil Count: {count}")
        
        temperature = self.temp_sensor.read_temperature()
        
        log_entry = self.logger.log_detection(count, temperature, self.is_after_mixing, activity)
        
        self.recommendation_label.setText(f"Recommendation: {log_entry.recommendation}")
        
        self.log_message(f"{log_entry.timestamp} - Count: {count}, Temp: {temperature:.1f}°C, Rec: {log_entry.recommendation}")
        
        if count > 0 or self.is_after_mixing:
            self.email_notifier.send_detection_alert(
                log_entry.timestamp,
                count,
                temperature,
                log_entry.recommendation,
                activity
            )

    def update_temperature(self):
        temperature = self.temp_sensor.read_temperature()
        if temperature is not None:
            self.temp_label.setText(f"Temperature: {temperature:.1f}°C")

    def log_message(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        if self.is_scanning:
            self.stop_scan()
        self.led_controller.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
