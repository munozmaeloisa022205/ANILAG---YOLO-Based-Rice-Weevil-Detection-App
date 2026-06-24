#!/usr/bin/env python3
"""
Anilag - Rice Weevil Detection System
Demo version to show GUI without ML dependencies
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QTextEdit, QGridLayout, QGroupBox,
                             QTabWidget, QListWidget, QListWidgetItem, QComboBox, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush
from typing import Optional, List
import threading
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv


# ============================================================================
# MOCK DETECTION MODULE
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


class MockDetector:
    def __init__(self):
        self.initialized = False

    def initialize(self) -> bool:
        self.initialized = True
        print("Mock detector initialized (demo mode)")
        return True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        # Simulate random detections for demo
        import random
        if random.random() > 0.7:
            h, w = frame.shape[:2]
            boxes = [[int(w*0.3), int(h*0.3), int(w*0.5), int(h*0.5)]]
            confidences = [0.85]
            class_ids = [0]
            class_names = ["rice_weevil"]
            return DetectionResult(boxes, confidences, class_ids, class_names)
        return DetectionResult([], [], [], ["rice_weevil"])

    def draw_detections(self, frame: np.ndarray, detection: DetectionResult, color=(0, 255, 0)) -> np.ndarray:
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
# HARDWARE MODULES (MOCK)
# ============================================================================

class Camera:
    def __init__(self, camera_id: int = 0, camera_name: str = "Camera", width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        self.running = False
        self.frame_callback: Optional = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.current_frame = None

    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                # Create a blank frame if no camera
                self.current_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.current_frame[:] = (50, 50, 50)
                cv2.putText(self.current_frame, "No Camera - Demo Mode", 
                           (self.width//2 - 150, self.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return True
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            return True
        except Exception as e:
            print(f"Camera initialization error: {e}")
            # Create blank frame for demo
            self.current_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.current_frame[:] = (50, 50, 50)
            cv2.putText(self.current_frame, "Demo Mode", 
                       (self.width//2 - 50, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return True

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
        import random
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        with self.lock:
                            self.current_frame = frame.copy()
                            if self.recording and self.video_writer:
                                self.video_writer.write(frame)
                        if self.frame_callback:
                            self.frame_callback(frame)
                else:
                    # Generate demo frame
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    frame[:] = (30 + random.randint(0, 20), 30 + random.randint(0, 20), 30 + random.randint(0, 20))
                    cv2.putText(frame, f"{self.camera_name}", 
                               (self.width//2 - 80, self.height//2 - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, "Anilag Demo Mode", 
                               (self.width//2 - 100, self.height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.putText(frame, "Camera Simulation", 
                               (self.width//2 - 80, self.height//2 + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
                    with self.lock:
                        self.current_frame = frame.copy()
                        if self.recording and self.video_writer:
                            self.video_writer.write(frame)
                    if self.frame_callback:
                        self.frame_callback(frame)
            except Exception as e:
                print(f"Capture loop error: {e}")
                break
            self.threading_event_wait()

    def threading_event_wait(self):
        import time
        time.sleep(0.05)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def start_recording(self, output_path: str):
        if not self.running:
            return False
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        self.recording = True
        print(f"Recording started: {output_path}")
        return True

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        print("Recording stopped")

    def stop(self):
        self.stop_recording()
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
        self.initialized = False
        self.demo_temp = 25.0

    def initialize(self) -> bool:
        self.initialized = True
        print("Temperature sensor initialized (demo mode)")
        return True

    def read_temperature(self) -> Optional[float]:
        import random
        self.demo_temp = 25.0 + random.uniform(-2, 5)
        return self.demo_temp


class LEDController:
    def __init__(self, gpio_pin: int = 18, led_count: int = 60, brightness: int = 255):
        self.gpio_pin = gpio_pin
        self.led_count = led_count
        self.brightness = brightness
        self.initialized = False
        self.current_color = "off"

    def initialize(self) -> bool:
        self.initialized = True
        print("LED controller initialized (demo mode)")
        return True

    def set_color(self, r: int, g: int, b: int):
        if r == 255 and g == 0 and b == 0:
            self.current_color = "red"
        elif r == 255 and g == 255 and b == 255:
            self.current_color = "white"
        elif r == 0 and g == 0 and b == 0:
            self.current_color = "off"
        print(f"Demo: LEDs set to RGB({r}, {g}, {b}) - {self.current_color}")

    def set_red(self):
        self.set_color(255, 0, 0)

    def set_white(self):
        self.set_color(255, 255, 255)

    def off(self):
        self.set_color(0, 0, 0)

    def cleanup(self):
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
            return True  # Continue anyway in demo

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
        return log_entry


# ============================================================================
# NOTIFICATION MODULE (MOCK)
# ============================================================================

class EmailNotifier:
    def __init__(self, config_file: str = 'config.env'):
        load_dotenv(config_file)
        self.enabled = False

    def initialize(self) -> bool:
        print("Email notifier disabled (demo mode)")
        return False

    def send_email(self, subject: str, body: str, is_html: bool = False) -> bool:
        print(f"Demo: Would send email - {subject}")
        return True

    def send_detection_alert(self, timestamp: str, rice_weevil_count: int, 
                            temperature: Optional[float], recommendation: str, 
                            activity: str) -> bool:
        print(f"Demo: Detection alert - {recommendation}")
        return True

    def send_activity_log(self, activity: str, details: str) -> bool:
        print(f"Demo: Activity log - {activity}")
        return True


# ============================================================================
# GUI MODULE
# ============================================================================

class RiceGrainLogo(QWidget):
    def __init__(self, size=200):
        super().__init__()
        self.setFixedSize(size, size)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Draw rice grain (larger, more detailed)
        grain_color = QColor(210, 180, 140)  # Light brown rice
        painter.setPen(QPen(QColor(160, 130, 90), 2))
        painter.setBrush(QBrush(grain_color))
        
        # Draw pointed grain shape using path
        from PyQt5.QtGui import QPainterPath
        grain_path = QPainterPath()
        grain_path.moveTo(center_x - 50, center_y)
        grain_path.quadTo(center_x - 30, center_y - 25, center_x + 10, center_y - 20)
        grain_path.quadTo(center_x + 50, center_y - 10, center_x + 55, center_y)
        grain_path.quadTo(center_x + 50, center_y + 10, center_x + 10, center_y + 20)
        grain_path.quadTo(center_x - 30, center_y + 25, center_x - 50, center_y)
        painter.drawPath(grain_path)
        
        # Add grain texture lines
        painter.setPen(QPen(QColor(180, 150, 110), 1))
        for i in range(-40, 45, 20):
            painter.drawLine(center_x + i, center_y - 12, center_x + i, center_y + 12)
        
        # Draw stem/leaf
        painter.setPen(QPen(QColor(34, 139, 34), 3))
        painter.setBrush(QBrush(QColor(46, 139, 87)))
        painter.drawLine(center_x - 50, center_y, center_x - 60, center_y - 30)
        # Small leaf
        painter.drawEllipse(center_x - 65, center_y - 40, 15, 8)
        
        # Draw weevil (small beetle on the grain)
        weevil_x = center_x + 20
        weevil_y = center_y - 5
        
        # Weevil body (dark brown)
        weevil_color = QColor(80, 50, 30)
        painter.setPen(QPen(QColor(60, 40, 20), 1))
        painter.setBrush(QBrush(weevil_color))
        
        # Weevil body (oval)
        painter.drawEllipse(weevil_x - 8, weevil_y - 4, 16, 8)
        
        # Weevil head
        painter.drawEllipse(weevil_x + 6, weevil_y - 3, 6, 6)
        
        # Weevil snout (rostrum)
        painter.setPen(QPen(QColor(60, 40, 20), 2))
        painter.drawLine(weevil_x + 12, weevil_y, weevil_x + 18, weevil_y)
        
        # Weevil antennae
        painter.setPen(QPen(QColor(60, 40, 20), 1))
        painter.drawLine(weevil_x + 8, weevil_y - 3, weevil_x + 12, weevil_y - 8)
        painter.drawLine(weevil_x + 8, weevil_y + 3, weevil_x + 12, weevil_y + 8)
        
        # Weevil legs (6 legs)
        painter.setPen(QPen(QColor(60, 40, 20), 1))
        # Left legs
        painter.drawLine(weevil_x - 2, weevil_y - 2, weevil_x - 6, weevil_y - 8)
        painter.drawLine(weevil_x + 2, weevil_y - 2, weevil_x + 2, weevil_y - 10)
        painter.drawLine(weevil_x + 6, weevil_y - 2, weevil_x + 10, weevil_y - 8)
        # Right legs
        painter.drawLine(weevil_x - 2, weevil_y + 2, weevil_x - 6, weevil_y + 8)
        painter.drawLine(weevil_x + 2, weevil_y + 2, weevil_x + 2, weevil_y + 10)
        painter.drawLine(weevil_x + 6, weevil_y + 2, weevil_x + 10, weevil_y + 8)
        
        # Detection indicator (red circle around weevil)
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(weevil_x, weevil_y, 25, 25)
        
        painter.end()


class MenuPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        # Logo
        self.logo = RiceGrainLogo(size=150)
        logo_container = QWidget()
        logo_layout = QVBoxLayout()
        logo_layout.addWidget(self.logo)
        logo_layout.setAlignment(Qt.AlignCenter)
        logo_container.setLayout(logo_layout)
        layout.addWidget(logo_container)
        
        # Title
        title_label = QLabel("ANILAG")
        title_label.setFont(QFont("Arial", 48, QFont.Bold))
        title_label.setStyleSheet("color: #2E7D32; margin: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Rice Weevil Detection System")
        subtitle_label.setFont(QFont("Arial", 18))
        subtitle_label.setStyleSheet("color: #555; margin-bottom: 40px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_label)
        
        # Start Scan button
        self.start_button = QPushButton("START SCAN")
        self.start_button.setMinimumSize(200, 60)
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 30px;
                border: 3px solid #388E3C;
            }
            QPushButton:hover {
                background-color: #66BB6A;
                border: 3px solid #43A047;
            }
            QPushButton:pressed {
                background-color: #43A047;
            }
        """)
        self.start_button.clicked.connect(self.on_start_scan)
        layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        
        # Info text
        info_label = QLabel("Dual Camera Detection System\nRaspberry Pi 5 Optimized")
        info_label.setFont(QFont("Arial", 12))
        info_label.setStyleSheet("color: #888; margin-top: 40px;")
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        self.setLayout(layout)
    
    def on_start_scan(self):
        if self.parent:
            self.parent.go_to_detection_page()

class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str)

    def __init__(self, camera_left: Camera, camera_right: Camera, detector: MockDetector):
        super().__init__()
        self.camera_left = camera_left
        self.camera_right = camera_right
        self.detector = detector
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            frame_left = self.camera_left.get_frame()
            frame_right = self.camera_right.get_frame()
            
            if frame_left is not None:
                detection_left = self.detector.detect(frame_left)
                self.frame_ready_left.emit(frame_left, detection_left)
            
            if frame_right is not None:
                detection_right = self.detector.detect(frame_right)
                self.frame_ready_right.emit(frame_right, detection_right)
            
            # Use left camera for detection updates
            if frame_left is not None:
                detection = self.detector.detect(frame_left)
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
        self.setWindowTitle("Anilag - Rice Weevil Detection System (DEMO)")
        self.setGeometry(100, 100, 1400, 900)
        
        load_dotenv('config.env')
        
        # Create two camera instances
        self.camera_left = Camera(
            camera_id=int(os.getenv('CAMERA_ID_LEFT', '0')),
            camera_name="LEFT CAMERA",
            width=int(os.getenv('CAMERA_WIDTH', '640')),
            height=int(os.getenv('CAMERA_HEIGHT', '480')),
            fps=int(os.getenv('CAMERA_FPS', '30'))
        )
        
        self.camera_right = Camera(
            camera_id=int(os.getenv('CAMERA_ID_RIGHT', '1')),
            camera_name="RIGHT CAMERA",
            width=int(os.getenv('CAMERA_WIDTH', '640')),
            height=int(os.getenv('CAMERA_HEIGHT', '480')),
            fps=int(os.getenv('CAMERA_FPS', '30'))
        )
        
        self.active_camera = "left"  # 'left' or 'right'
        self.detector = MockDetector()
        self.temp_sensor = TemperatureSensor()
        self.led_controller = LEDController()
        self.logger = DetectionLogger(log_file=os.getenv('LOG_FILE', 'logs/detection_log.csv'))
        self.email_notifier = EmailNotifier('config.env')
        
        self.detection_thread: Optional[DetectionThread] = None
        self.is_scanning = False
        self.is_after_mixing = False
        self.previous_scans = []  # List of recorded video paths
        self.recordings_dir = 'recordings'
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Create stacked widget for page navigation
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create pages
        self.menu_page = MenuPage(self)
        self.detection_page = QWidget()
        
        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.detection_page)
        
        # Initialize detection page UI
        self.init_detection_ui()
        self.initialize_components()
        
        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)

    def init_detection_ui(self):
        central_widget = QWidget()
        self.detection_page.setLayout(QVBoxLayout())
        self.detection_page.layout().addWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Camera feeds
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=2)
        
        # Camera selection tabs
        self.camera_tabs = QTabWidget()
        
        # Left camera tab
        self.left_camera_widget = QWidget()
        left_cam_layout = QVBoxLayout()
        left_cam_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label_left = QLabel()
        self.camera_label_left.setMinimumSize(640, 480)
        self.camera_label_left.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.camera_label_left.setAlignment(Qt.AlignCenter)
        self.camera_label_left.setText("LEFT CAMERA - Click Start Scan")
        left_cam_layout.addWidget(self.camera_label_left)
        
        self.left_camera_widget.setLayout(left_cam_layout)
        self.camera_tabs.addTab(self.left_camera_widget, "Left Camera")
        
        # Right camera tab
        self.right_camera_widget = QWidget()
        right_cam_layout = QVBoxLayout()
        right_cam_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_label_right = QLabel()
        self.camera_label_right.setMinimumSize(640, 480)
        self.camera_label_right.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.camera_label_right.setAlignment(Qt.AlignCenter)
        self.camera_label_right.setText("RIGHT CAMERA - Click Start Scan")
        right_cam_layout.addWidget(self.camera_label_right)
        
        self.right_camera_widget.setLayout(right_cam_layout)
        self.camera_tabs.addTab(self.right_camera_widget, "Right Camera")
        
        self.camera_tabs.currentChanged.connect(self.on_camera_tab_changed)
        left_panel.addWidget(self.camera_tabs)
        
        # Detection info
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
        
        # Right panel - Controls and Previous Scans
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=1)
        
        # Scan controls
        scan_group = QGroupBox("Scan Controls")
        scan_layout = QVBoxLayout()
        
        self.stop_button = QPushButton("Stop Scan")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setStyleSheet("""
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
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)  # Disabled until scan starts
        scan_layout.addWidget(self.stop_button)
        
        self.back_button = QPushButton("Back to Menu")
        self.back_button.setMinimumHeight(40)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        self.back_button.clicked.connect(self.go_to_menu_page)
        scan_layout.addWidget(self.back_button)
        
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
        
        # LED controls
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
        
        # Previous Scans section
        scans_group = QGroupBox("Previous Scans")
        scans_layout = QVBoxLayout()
        
        self.scans_list = QListWidget()
        self.scans_list.setMaximumHeight(150)
        self.scans_list.itemDoubleClicked.connect(self.play_selected_scan)
        scans_layout.addWidget(self.scans_list)
        
        scans_group.setLayout(scans_layout)
        right_panel.addWidget(scans_group)
        
        # Detection log
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        self.status_label = QLabel("Ready - DEMO MODE")
        self.statusBar().addWidget(self.status_label)
    
    def go_to_detection_page(self):
        self.stacked_widget.setCurrentIndex(1)
        self.start_scan()
    
    def go_to_menu_page(self):
        if self.is_scanning:
            self.stop_scan()
        self.stacked_widget.setCurrentIndex(0)

    def initialize_components(self):
        self.detector.initialize()
        self.temp_sensor.initialize()
        self.led_controller.initialize()
        self.logger.initialize()
        self.email_notifier.initialize()
        self.log_message("Anilag Demo Mode Initialized")
        self.log_message("All components running in simulation mode")

    def start_scan(self):
        if not self.camera_left.start() or not self.camera_right.start():
            self.log_message("Failed to start cameras")
            return
        
        # Start recording both cameras
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_left_path = os.path.join(self.recordings_dir, f"scan_left_{timestamp}.mp4")
        self.recording_right_path = os.path.join(self.recordings_dir, f"scan_right_{timestamp}.mp4")
        
        self.camera_left.start_recording(self.recording_left_path)
        self.camera_right.start_recording(self.recording_right_path)
        
        self.detection_thread = DetectionThread(self.camera_left, self.camera_right, self.detector)
        self.detection_thread.frame_ready_left.connect(self.update_frame_left)
        self.detection_thread.frame_ready_right.connect(self.update_frame_right)
        self.detection_thread.detection_update.connect(self.update_detection)
        self.detection_thread.start()
        
        self.is_scanning = True
        self.stop_button.setEnabled(True)
        self.status_label.setText("Scanning... (DEMO)")
        self.log_message("Scan started (demo mode)")
        self.log_message("Recording both cameras...")
        self.email_notifier.send_activity_log("Scan Started", "Live detection has been initiated")

    def stop_scan(self):
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        # Stop recording and save to previous scans
        self.camera_left.stop_recording()
        self.camera_right.stop_recording()
        
        # Add to previous scans list
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scan_item = f"{timestamp} - Left & Right Cameras"
        self.previous_scans.append({
            'name': scan_item,
            'left_path': self.recording_left_path,
            'right_path': self.recording_right_path,
            'timestamp': timestamp
        })
        self.scans_list.addItem(scan_item)
        
        self.camera_left.stop()
        self.camera_right.stop()
        
        self.is_scanning = False
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready - DEMO MODE")
        self.log_message("Scan stopped")
        self.log_message("Recording saved to previous scans")
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
        self.log_message("Red light activated (demo)")
        self.email_notifier.send_activity_log("LED Control", "Red light activated to lure rice weevils")

    def set_white_light(self):
        self.led_controller.set_white()
        self.log_message("White light activated (demo)")
        self.email_notifier.send_activity_log("LED Control", "White light activated for detection")

    def set_leds_off(self):
        self.led_controller.off()
        self.log_message("LEDs turned off (demo)")
        self.email_notifier.send_activity_log("LED Control", "LEDs turned off")

    def on_camera_tab_changed(self, index):
        if index == 0:
            self.active_camera = "left"
            self.status_label.setText("Viewing LEFT CAMERA - DEMO MODE")
        else:
            self.active_camera = "right"
            self.status_label.setText("Viewing RIGHT CAMERA - DEMO MODE")
    
    def update_frame_left(self, frame: np.ndarray, detection: DetectionResult):
        annotated_frame = self.detector.draw_detections(frame, detection)
        
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label_left.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label_left.setPixmap(scaled_pixmap)
    
    def update_frame_right(self, frame: np.ndarray, detection: DetectionResult):
        annotated_frame = self.detector.draw_detections(frame, detection)
        
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label_right.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label_right.setPixmap(scaled_pixmap)

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

    def play_selected_scan(self, item):
        # In a full implementation, this would play the video
        # For demo, just show a message
        scan_index = self.scans_list.row(item)
        if scan_index < len(self.previous_scans):
            scan_info = self.previous_scans[scan_index]
            self.log_message(f"Playing scan: {scan_info['name']}")
            self.log_message(f"Left: {scan_info['left_path']}")
            self.log_message(f"Right: {scan_info['right_path']}")
            # In production, use cv2.VideoCapture to play the video
    
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
