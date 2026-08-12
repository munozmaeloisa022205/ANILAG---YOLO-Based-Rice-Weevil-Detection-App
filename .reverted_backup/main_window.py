import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QTextEdit, QGridLayout, QGroupBox,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QListWidget, QListWidgetItem, QSplitter, QMessageBox,
                             QFileDialog, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from typing import Optional
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import modules
from src.hardware.camera import Camera, DualCameraManager
from src.hardware.temperature import TemperatureSensor
from src.hardware.led_controller import LEDController
from src.detection.yolov11_detector import YOLOv11Detector, DetectionResult
from src.logging.logger import DetectionLogger
from src.notification.email_notifier import EmailNotifier
from src.backend.database import get_database
from src.backend import report_builder


class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str)
    stats_update = pyqtSignal(float, float)  # detections per second, avg inference ms

    def __init__(self, camera_manager: DualCameraManager, detector: YOLOv11Detector,
                 interval_ms: int = 200):
        super().__init__()
        self.camera_manager = camera_manager
        self.detector = detector
        # Minimum time between detection cycles. YOLOv11n on the Pi 5 ARM CPU needs
        # ~20ms (NCNN) to ~120ms (PyTorch) per frame, so a fixed 20 FPS loop would
        # saturate all four cores and starve the GUI thread.
        self.interval_ms = max(0, interval_ms)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            cycle_started = time.perf_counter()
            left_frame = self.camera_manager.get_left_frame()
            right_frame = self.camera_manager.get_right_frame()

            # Emit combined detection update
            total_count = 0
            confidences = []
            if left_frame is not None:
                detection_left = self.detector.detect(left_frame)
                total_count += detection_left.count
                confidences.extend(detection_left.confidences)
                self.frame_ready_left.emit(left_frame, detection_left)

            if right_frame is not None:
                detection_right = self.detector.detect(right_frame)
                total_count += detection_right.count
                confidences.extend(detection_right.confidences)
                self.frame_ready_right.emit(right_frame, detection_right)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            self.detection_update.emit(total_count, avg_confidence, "Detection")

            elapsed_ms = (time.perf_counter() - cycle_started) * 1000
            self.stats_update.emit(1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
                                   self.detector.avg_inference_ms)

            # Throttle so the Pi 5 keeps headroom for the GUI, recording and DB writes
            remaining = self.interval_ms - elapsed_ms
            self.msleep(int(remaining) if remaining > 0 else 1)

    def stop(self):
        self.running = False
        self.wait()


class EmailThread(QThread):
    """Runs an EmailNotifier call off the UI thread so SMTP never freezes the GUI."""
    finished_with_status = pyqtSignal(bool, str)

    def __init__(self, description: str, send_func, *args, **kwargs):
        super().__init__()
        self.description = description
        self.send_func = send_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        success = self.send_func(*self.args, **self.kwargs)
        if success:
            self.finished_with_status.emit(True, f"Email sent: {self.description}")
        else:
            self.finished_with_status.emit(False, f"Email failed: {self.description} (see console for details)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anilag - Rice Weevil Detection System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Load configuration
        load_dotenv('config.env')
        
        # Initialize components
        self.camera_manager = DualCameraManager(
            left_camera_id=int(os.getenv('LEFT_CAMERA_ID', '0')),
            right_camera_id=int(os.getenv('RIGHT_CAMERA_ID', '1')),
            width=int(os.getenv('CAMERA_WIDTH', '640')),
            height=int(os.getenv('CAMERA_HEIGHT', '480')),
            fps=int(os.getenv('CAMERA_FPS', '30'))
        )
        
        self.detector = YOLOv11Detector(
            model_path=os.getenv('MODEL_PATH', 'models/sitophilus_oryzae_v2-3_best.pt'),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
            iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.7'))
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
        
        # Initialize database - use external SSD path if configured
        _default_db = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'anilag.db')
        _db_storage = os.getenv('DB_STORAGE_PATH', '').strip()
        _db_path = os.path.join(_db_storage, 'anilag.db') if _db_storage else _default_db
        self.db = get_database(_db_path)
        
        self.detection_thread: Optional[DetectionThread] = None
        self.is_scanning = False
        self.is_after_mixing = False
        
        # Detection data for table
        self.detection_data = []  # List of tuples: (timestamp, count, temperature, recommendation)
        
        # Track last log time for 1-minute interval logging
        self.last_log_time = None
        
        # Recording infrastructure
        self.video_writer_left = None
        self.video_writer_right = None
        self.current_scan_folder = None
        self.current_scan_id = None
        _default_scans_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'previous_scans')
        _scan_storage = os.getenv('SCAN_STORAGE_PATH', '').strip()
        self.previous_scans_dir = os.path.join(_scan_storage, 'previous_scans') if _scan_storage else _default_scans_dir
        os.makedirs(self.previous_scans_dir, exist_ok=True)
        
        # Scan metadata
        self.scan_start_time = None
        self.scan_max_count = 0
        self.scan_avg_temp = 0.0
        self.scan_temp_readings = []
        
        # Image capture settings
        self.high_weevil_threshold = int(os.getenv('HIGH_WEEVIL_THRESHOLD', '5'))
        self.last_image_capture_time = None
        self.image_capture_cooldown = int(os.getenv('IMAGE_CAPTURE_COOLDOWN', '30'))  # seconds between captures
        self.latest_annotated_left = None
        self.latest_annotated_right = None
        
        # Scan protocol: each scan runs for a fixed duration, then auto-stops and emails the report
        self.scan_duration_seconds = int(os.getenv('SCAN_DURATION_SECONDS', '180'))
        self.scan_images_dir = None
        self.email_threads = []
        # Sparse baseline log interval - guarantees a scan always has log rows, even when
        # nothing significant is found, so "no weevils" is recorded rather than missing.
        self.log_interval_seconds = int(os.getenv('LOG_INTERVAL_SECONDS', '60'))
        # Minimum gap between log rows triggered by a high weevil count
        self.significant_log_interval_seconds = int(os.getenv('SIGNIFICANT_LOG_INTERVAL_SECONDS', '5'))
        self.last_significant_log_time = None
        # Detection cycle interval - throttles YOLO inference on the Pi 5 CPU
        self.detection_interval_ms = int(os.getenv('DETECTION_INTERVAL_MS', '200'))
        self.scan_detection_count = 0
        self.scan_image_count = 0
        self.reports_dir = os.path.join(self.previous_scans_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup UI
        self.init_ui()
        self.initialize_components()
        
        # Setup temperature update timer
        self.temp_timer = QTimer()
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(1000)  # Update every second
        
        # Setup clock update timer
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  # Update every second
        
        # Scan protocol timer - stops the scan after the configured duration
        self.scan_timer = QTimer()
        self.scan_timer.setSingleShot(True)
        self.scan_timer.timeout.connect(self.on_scan_duration_elapsed)
        
        # Countdown timer for the remaining scan time shown in the status label
        self.scan_countdown_timer = QTimer()
        self.scan_countdown_timer.timeout.connect(self.update_scan_countdown)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create header with logo
        header_widget = QWidget()
        header_layout = QHBoxLayout()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #f5f5f5; border-bottom: 2px solid #ddd;")
        header_widget.setMaximumHeight(80)
        
        # Logo label
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("ANILAG")
            logo_label.setFont(QFont("Arial", 20, QFont.Bold))
            logo_label.setStyleSheet("color: #2E7D32;")
        header_layout.addWidget(logo_label)
        
        # Title and tagline container
        title_container = QWidget()
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_container.setLayout(title_layout)
        
        # Title label
        title_label = QLabel("Anilag")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setStyleSheet("color: #2E7D32;")
        title_layout.addWidget(title_label)
        
        # Tagline label
        tagline_label = QLabel("Rice Weevil Detection System")
        tagline_label.setFont(QFont("Arial", 11))
        tagline_label.setStyleSheet("color: #666;")
        title_layout.addWidget(tagline_label)
        
        header_layout.addWidget(title_container)
        
        header_layout.addStretch()
        main_layout.addWidget(header_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create Live Feed tab
        self.live_feed_tab = QWidget()
        self.setup_live_feed_tab()
        self.tab_widget.addTab(self.live_feed_tab, "Live Feed")
        
        # Create Detection Information tab
        self.detection_info_tab = QWidget()
        self.setup_detection_info_tab()
        self.tab_widget.addTab(self.detection_info_tab, "Detection Information")
        
        # Create Scan History tab (reads scans, logs and images back from the database)
        self.scan_history_tab = QWidget()
        self.setup_scan_history_tab()
        self.tab_widget.addTab(self.scan_history_tab, "Scan History")
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

    def setup_live_feed_tab(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.live_feed_tab.setLayout(layout)
        
        # Left panel - Camera feeds (large) and controls below
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        layout.addLayout(left_panel, stretch=3)
        
        # Camera feeds container - larger
        camera_container = QHBoxLayout()
        camera_container.setSpacing(10)
        left_panel.addLayout(camera_container, stretch=3)
        
        # Left camera feed label - larger
        self.left_camera_label = QLabel()
        self.left_camera_label.setMinimumSize(320, 240)
        self.left_camera_label.setSizePolicy(self.left_camera_label.sizePolicy().horizontalPolicy(), self.left_camera_label.sizePolicy().verticalPolicy())
        self.left_camera_label.setStyleSheet("border: 2px solid #333; background-color: #000; border-radius: 5px;")
        self.left_camera_label.setAlignment(Qt.AlignCenter)
        self.left_camera_label.setText("No Signal")
        camera_container.addWidget(self.left_camera_label, stretch=1)
        
        # Add label overlay on top of left camera
        left_camera_title = QLabel("Left Camera")
        left_camera_title.setFont(QFont("Arial", 12, QFont.Bold))
        left_camera_title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 3px;")
        left_camera_title.setAlignment(Qt.AlignCenter)
        left_camera_title.setParent(self.left_camera_label)
        left_camera_title.move(10, 10)
        left_camera_title.show()
        
        # Right camera feed label - larger
        self.right_camera_label = QLabel()
        self.right_camera_label.setMinimumSize(320, 240)
        self.right_camera_label.setSizePolicy(self.right_camera_label.sizePolicy().horizontalPolicy(), self.right_camera_label.sizePolicy().verticalPolicy())
        self.right_camera_label.setStyleSheet("border: 2px solid #333; background-color: #000; border-radius: 5px;")
        self.right_camera_label.setAlignment(Qt.AlignCenter)
        self.right_camera_label.setText("No Signal")
        camera_container.addWidget(self.right_camera_label, stretch=1)
        
        # Add label overlay on top of right camera
        right_camera_title = QLabel("Right Camera")
        right_camera_title.setFont(QFont("Arial", 12, QFont.Bold))
        right_camera_title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 5px; border-radius: 3px;")
        right_camera_title.setAlignment(Qt.AlignCenter)
        right_camera_title.setParent(self.right_camera_label)
        right_camera_title.move(10, 10)
        right_camera_title.show()
        
        # Scan controls below camera feeds
        scan_group = QGroupBox("Scan Controls")
        scan_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        scan_layout = QVBoxLayout()
        scan_layout.setSpacing(8)
        
        self.start_button = QPushButton("Start Scan")
        self.start_button.setMinimumHeight(45)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 8px;
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
        
        self.view_scans_button = QPushButton("View Previous Scans")
        self.view_scans_button.setMinimumHeight(40)
        self.view_scans_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 13px;
                border-radius: 5px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.view_scans_button.clicked.connect(self.view_previous_scans)
        scan_layout.addWidget(self.view_scans_button)

        self.mixing_button = QPushButton("Mark as After Mixing/Sifting")
        self.mixing_button.setMinimumHeight(40)
        self.mixing_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 13px;
                border-radius: 5px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.mixing_button.clicked.connect(self.toggle_mixing_state)
        scan_layout.addWidget(self.mixing_button)

        scan_group.setLayout(scan_layout)
        left_panel.addWidget(scan_group)
        
        # LED controls below scan controls
        led_group = QGroupBox("LED Controls")
        led_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        led_layout = QHBoxLayout()
        led_layout.setSpacing(8)
        
        self.red_light_button = QPushButton("Red Light")
        self.red_light_button.setMinimumHeight(35)
        self.red_light_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 13px;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.red_light_button.clicked.connect(self.set_red_light)
        led_layout.addWidget(self.red_light_button)
        
        self.white_light_button = QPushButton("White Light")
        self.white_light_button.setMinimumHeight(35)
        self.white_light_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: black;
                font-size: 13px;
                border-radius: 5px;
                border: 2px solid #333;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.white_light_button.clicked.connect(self.set_white_light)
        led_layout.addWidget(self.white_light_button)
        
        self.led_off_button = QPushButton("LEDs Off")
        self.led_off_button.setMinimumHeight(35)
        self.led_off_button.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                font-size: 13px;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.led_off_button.clicked.connect(self.set_leds_off)
        led_layout.addWidget(self.led_off_button)
        
        led_group.setLayout(led_layout)
        left_panel.addWidget(led_group)
        
        # Right panel - Current detection info and detection log
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        layout.addLayout(right_panel, stretch=1)
        
        # Current detection info
        current_info_group = QGroupBox("Current Detection")
        current_info_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        current_info_layout = QVBoxLayout()
        current_info_layout.setSpacing(8)
        current_info_layout.setContentsMargins(10, 10, 10, 10)
        
        self.count_label = QLabel("Weevil Count: 0")
        self.count_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.count_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.count_label)
        
        self.confidence_label = QLabel("Avg Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 11))
        self.confidence_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.confidence_label)
        
        self.temp_label = QLabel("Temperature: --°C")
        self.temp_label.setFont(QFont("Arial", 12))
        self.temp_label.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 5px; border: 1px solid #bbdefb;")
        current_info_layout.addWidget(self.temp_label)
        
        self.recommendation_label = QLabel("Recommendation: --")
        self.recommendation_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.recommendation_label.setStyleSheet("color: #0066cc; padding: 8px; background-color: #fff3e0; border-radius: 5px; border: 1px solid #ffe0b2;")
        current_info_layout.addWidget(self.recommendation_label)
        
        # Time and date display
        self.date_label = QLabel()
        self.date_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.date_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        self.date_label.setAlignment(Qt.AlignCenter)
        current_info_layout.addWidget(self.date_label)
        
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.time_label.setStyleSheet("padding: 8px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        self.time_label.setAlignment(Qt.AlignCenter)
        current_info_layout.addWidget(self.time_label)
        
        current_info_group.setLayout(current_info_layout)
        right_panel.addWidget(current_info_group)
        
        # YOLOv11n model and inference performance
        model_group = QGroupBox("Detection Model")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(6)
        model_layout.setContentsMargins(10, 10, 10, 10)
        
        self.model_label = QLabel("Model: loading...")
        self.model_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("padding: 6px; background-color: #ede7f6; border-radius: 5px; border: 1px solid #d1c4e9;")
        model_layout.addWidget(self.model_label)
        
        self.model_detail_label = QLabel("Backend: --")
        self.model_detail_label.setFont(QFont("Arial", 9))
        self.model_detail_label.setWordWrap(True)
        self.model_detail_label.setStyleSheet("padding: 6px; background-color: #f3e5f5; border-radius: 5px; border: 1px solid #e1bee7; color: #444;")
        model_layout.addWidget(self.model_detail_label)
        
        self.performance_label = QLabel("Inference: -- ms   |   Rate: -- /s")
        self.performance_label.setFont(QFont("Consolas", 9))
        self.performance_label.setStyleSheet("padding: 6px; background-color: #fff8e1; border-radius: 5px; border: 1px solid #ffecb3;")
        model_layout.addWidget(self.performance_label)
        
        self.model_warning_label = QLabel()
        self.model_warning_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.model_warning_label.setWordWrap(True)
        self.model_warning_label.setStyleSheet("padding: 6px; background-color: #ffebee; border-radius: 5px; border: 1px solid #ef9a9a; color: #b71c1c;")
        self.model_warning_label.setVisible(False)
        model_layout.addWidget(self.model_warning_label)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Detection log display - compact
        log_group = QGroupBox("Detection Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;")
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

    def setup_detection_info_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.detection_info_tab.setLayout(layout)
        
        # Detection history table
        table_group = QGroupBox("Detection History")
        table_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(5, 5, 5, 5)
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(["Date and Timestamp", "Count", "Temperature", "Recommendation"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detection_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detection_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
                gridline-color: #eee;
            }
            QTableWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: 1px solid #ddd;
                font-weight: bold;
                color: #333;
            }
        """)
        table_layout.addWidget(self.detection_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

    def setup_scan_history_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.scan_history_tab.setLayout(layout)
        
        table_style = """
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
                gridline-color: #eee;
            }
            QTableWidget::item { padding: 5px; border-bottom: 1px solid #eee; }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: 1px solid #ddd;
                font-weight: bold;
                color: #333;
            }
        """
        
        self.db_status_label = QLabel("Database: --")
        self.db_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.db_status_label.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 5px; border: 1px solid #bbdefb;")
        layout.addWidget(self.db_status_label)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)
        
        # Left: stored scans
        scans_group = QGroupBox("Stored Scans")
        scans_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        scans_layout = QVBoxLayout()
        scans_layout.setContentsMargins(5, 5, 5, 5)
        
        self.scan_table = QTableWidget()
        self.scan_table.setColumnCount(7)
        self.scan_table.setHorizontalHeaderLabels(
            ["Scan ID", "Start Time", "Max Count", "Avg Temp", "Log Rows", "Images", "Report"])
        self.scan_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.scan_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.scan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.scan_table.setSelectionMode(QTableWidget.SingleSelection)
        self.scan_table.setStyleSheet(table_style)
        self.scan_table.itemSelectionChanged.connect(self.on_scan_selected)
        scans_layout.addWidget(self.scan_table)
        
        scans_group.setLayout(scans_layout)
        splitter.addWidget(scans_group)
        
        # Right: images and detection log stored for the selected scan
        detail_group = QGroupBox("Stored Detection Log and Images")
        detail_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(5, 5, 5, 5)
        detail_layout.setSpacing(8)
        
        self.image_preview_label = QLabel("Select a scan to preview its stored images")
        self.image_preview_label.setMinimumHeight(200)
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("border: 2px solid #333; background-color: #000; color: #999; border-radius: 5px;")
        detail_layout.addWidget(self.image_preview_label, stretch=2)
        
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(120)
        self.image_list.setStyleSheet("font-family: Consolas, monospace; font-size: 10px; border: 1px solid #ddd; border-radius: 3px;")
        self.image_list.currentRowChanged.connect(self.on_history_image_selected)
        detail_layout.addWidget(self.image_list, stretch=1)
        
        self.history_log_table = QTableWidget()
        self.history_log_table.setColumnCount(4)
        self.history_log_table.setHorizontalHeaderLabels(["Timestamp", "Count", "Temp", "Recommendation"])
        self.history_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_log_table.setStyleSheet(table_style)
        detail_layout.addWidget(self.history_log_table, stretch=2)
        
        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)
        splitter.setSizes([600, 500])
        
        # Actions
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        
        def make_button(text, color, hover, handler):
            button = QPushButton(text)
            button.setMinimumHeight(38)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 5px;
                    padding: 6px;
                }}
                QPushButton:hover {{ background-color: {hover}; }}
                QPushButton:disabled {{ background-color: #bdbdbd; }}
            """)
            button.clicked.connect(handler)
            button_row.addWidget(button)
            return button
        
        self.refresh_history_button = make_button("Refresh", "#2196F3", "#1976D2", self.refresh_scan_history)
        self.resend_report_button = make_button("Email Selected Scan Report", "#4CAF50", "#45a049",
                                                self.resend_selected_report)
        self.export_report_button = make_button("Export Zip from Database", "#FF9800", "#F57C00",
                                                self.export_selected_report)
        self.open_scans_folder_button = make_button("Open Scans Folder", "#607D8B", "#455A64",
                                                   self.view_previous_scans)
        layout.addLayout(button_row)

    def initialize_components(self):
        # Initialize detector
        if self.detector.initialize():
            info = self.detector.get_model_info()
            self.log_message(f"YOLOv11n detector initialized: {info['model_name']} "
                             f"({info['backend']} backend, imgsz={info['imgsz']}, "
                             f"classes={', '.join(str(c) for c in info['classes']) or 'none'})")
        else:
            self.log_message("Failed to initialize YOLOv11 detector")
        self.update_model_display()
        
        # Initialize temperature sensor
        if self.temp_sensor.initialize():
            self.log_message("Temperature sensor initialized")
        else:
            self.log_message("Temperature sensor not available")
        
        # Initialize LED controller
        if self.led_controller.initialize():
            self.log_message("LED controller initialized")
        else:
            self.log_message("LED controller not available")
        
        # Initialize logger
        if self.logger.initialize():
            self.log_message("Logger initialized")
        else:
            self.log_message("Failed to initialize logger")
        
        # Initialize email notifier
        if self.email_notifier.initialize():
            self.log_message("Email notifier initialized")
        else:
            self.log_message("Email notifier disabled")
        
        # Populate the database-backed history view
        self.refresh_scan_history()

    def update_model_display(self):
        """Show which YOLOv11n weights and backend are actually loaded."""
        info = self.detector.get_model_info()
        status = "loaded" if info['initialized'] else "NOT LOADED"
        self.model_label.setText(f"{info['architecture']} - {info['model_name']} ({status})")
        classes = ', '.join(str(c) for c in info['classes']) or 'none'
        metrics = info['metrics']
        self.model_detail_label.setText(
            f"Backend: {info['backend']}   |   Device: {info['device']}   |   "
            f"Input: {info['imgsz']}px (trained {info['trained_imgsz']}px)\n"
            f"Classes: {classes}\n"
            f"Confidence >= {info['confidence_threshold']}   |   IoU {info['iou_threshold']}   |   "
            f"Cycle every {self.detection_interval_ms} ms\n"
            f"Training ({metrics['run']}, {metrics['epochs']} epochs): "
            f"mAP50 {metrics['mAP50']:.3f}   mAP50-95 {metrics['mAP50-95']:.3f}   "
            f"P {metrics['precision']:.3f}   R {metrics['recall']:.3f}")
        
        warning = info['warning']
        if not info['initialized']:
            warning = warning or "Model failed to load - no detections will be recorded."
        self.model_warning_label.setText(warning or "")
        self.model_warning_label.setVisible(bool(warning))
        if warning:
            self.log_message(f"MODEL WARNING: {warning}")

    def update_performance_stats(self, rate_per_second: float, avg_inference_ms: float):
        self.performance_label.setText(
            f"Inference: {avg_inference_ms:6.1f} ms   |   Rate: {rate_per_second:5.1f} /s")

    def toggle_scan(self):
        if self.is_scanning:
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self):
        if not self.camera_manager.start():
            self.log_message("Failed to start cameras")
            return
        
        # Create scan folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_scan_folder = os.path.join(self.previous_scans_dir, f"scan_{timestamp}")
        os.makedirs(self.current_scan_folder, exist_ok=True)
        
        # Dedicated folder for the captured images of detected rice weevils
        self.scan_images_dir = os.path.join(self.current_scan_folder, "detected_images")
        os.makedirs(self.scan_images_dir, exist_ok=True)
        
        # Generate scan ID for database
        self.current_scan_id = f"scan_{timestamp}"
        
        # Initialize video writers with compression
        width = int(os.getenv('CAMERA_WIDTH', '640'))
        height = int(os.getenv('CAMERA_HEIGHT', '480'))
        fps = int(os.getenv('CAMERA_FPS', '30'))
        
        left_video_path = os.path.join(self.current_scan_folder, "left_camera.mp4")
        right_video_path = os.path.join(self.current_scan_folder, "right_camera.mp4")

        # Use H.264 codec for better compression; fall back to mp4v if unavailable.
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.video_writer_left = cv2.VideoWriter(left_video_path, fourcc, fps, (width, height))
        if not self.video_writer_left.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer_left = cv2.VideoWriter(left_video_path, fourcc, fps, (width, height))

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.video_writer_right = cv2.VideoWriter(right_video_path, fourcc, fps, (width, height))
        if not self.video_writer_right.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer_right = cv2.VideoWriter(right_video_path, fourcc, fps, (width, height))
        
        # Reset scan metadata
        self.scan_start_time = datetime.now()
        self.scan_max_count = 0
        self.scan_avg_temp = 0.0
        self.scan_temp_readings = []
        self.scan_detection_count = 0
        self.scan_image_count = 0
        self.latest_annotated_left = None
        self.latest_annotated_right = None
        self.left_video_path = left_video_path
        self.right_video_path = right_video_path
        
        # Create scan record in database
        start_time_str = self.scan_start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.db.create_scan(self.current_scan_id, start_time_str, left_video_path, right_video_path)
        
        self.detection_thread = DetectionThread(self.camera_manager, self.detector,
                                                interval_ms=self.detection_interval_ms)
        self.detection_thread.frame_ready_left.connect(self.update_left_frame)
        self.detection_thread.frame_ready_right.connect(self.update_right_frame)
        self.detection_thread.detection_update.connect(self.update_detection)
        self.detection_thread.stats_update.connect(self.update_performance_stats)
        self.detection_thread.start()
        
        # Reset image/log tracking so the previous scan's timings do not leak in
        self.last_image_capture_time = None
        self.last_log_time = None
        self.last_significant_log_time = None
        
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
        # Start the fixed-duration scan protocol
        self.scan_timer.start(self.scan_duration_seconds * 1000)
        self.scan_countdown_timer.start(1000)
        self.update_scan_countdown()
        
        self.log_message(f"Scan started - Recording to {self.current_scan_folder}")
        self.log_message(f"Scan will run for {self.scan_duration_seconds} seconds, "
                         f"then the report will be emailed to {self.email_notifier.recipient_email or 'nobody (email disabled)'}")
        self.send_email_async("Scan Started", self.email_notifier.send_activity_log,
                              "Scan Started",
                              f"Live detection has been initiated. Scan ID: {self.current_scan_id}. "
                              f"Duration: {self.scan_duration_seconds} seconds.")

    def on_scan_duration_elapsed(self):
        """Called when the scan protocol duration is reached - auto-stop and email the report."""
        if not self.is_scanning:
            return
        self.log_message(f"Scan protocol duration ({self.scan_duration_seconds}s) reached - stopping scan")
        self.stop_scan()

    def update_scan_countdown(self):
        if not self.is_scanning:
            return
        remaining_ms = self.scan_timer.remainingTime()
        remaining = max(0, remaining_ms // 1000) if remaining_ms >= 0 else 0
        self.status_label.setText(f"Scanning... {remaining // 60:02d}:{remaining % 60:02d} remaining")

    def stop_scan(self):
        self.scan_timer.stop()
        self.scan_countdown_timer.stop()
        
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None
        
        self.camera_manager.stop()
        
        # Stop recording and save metadata
        if self.video_writer_left:
            self.video_writer_left.release()
            self.video_writer_left = None
        if self.video_writer_right:
            self.video_writer_right.release()
            self.video_writer_right = None
        
        # Persist the scan metadata; the detection log and images are already in the DB
        scan_id = self.current_scan_id
        if self.current_scan_folder and scan_id:
            self.save_scan_metadata()
        
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
        self.log_message("Scan stopped and saved")
        
        if scan_id:
            self.email_scan_report(scan_id)
            self.refresh_scan_history()

    def email_scan_report(self, scan_id: str, interactive: bool = False) -> Optional[str]:
        """Build the report archive from the database and email it.
        
        Everything in the archive - the detection log CSV and the captured rice weevil
        images - is read back out of SQLite, so this works for any past scan too.
        """
        video_paths = [p for p in (getattr(self, 'left_video_path', None),
                                   getattr(self, 'right_video_path', None)) if p]
        scan = self.db.get_scan_by_id(scan_id)
        if scan and not video_paths:
            video_paths = [p for p in (scan.get('left_video_path'), scan.get('right_video_path')) if p]
        
        zip_path = os.path.join(self.reports_dir, f"{scan_id}_report.zip")
        max_bytes = int(self.email_notifier.max_attachment_mb * 1024 * 1024)
        
        try:
            archive_path, info = report_builder.build_scan_archive(
                self.db, scan_id, zip_path, video_paths=video_paths, max_bytes=max_bytes)
        except Exception as e:
            self.log_message(f"Error building scan archive from database: {e}")
            return None
        
        if not archive_path:
            self.log_message(f"Could not build archive: {info.get('error')}")
            return None
        
        self.log_message(
            f"Scan archive built from database: {os.path.basename(archive_path)} "
            f"({info['archive_bytes'] / (1024 * 1024):.2f} MB, {info['log_entry_count']} log entries, "
            f"{info['image_count']} images, videos {'included' if info['included_videos'] else 'excluded'})")
        
        summary = report_builder.build_scan_summary(self.db, scan_id) or {}
        report_id = self.db.create_report(
            scan_id, self.email_notifier.recipient_email, info['archive_name'],
            info['archive_bytes'], info['image_count'], info['log_entry_count'],
            info['included_videos'])
        
        if not self.email_notifier.enabled:
            self.db.update_report_status(report_id, 'skipped', 'email disabled or not configured')
            self.log_message("Email disabled - archive saved locally and recorded in the database")
            if interactive:
                QMessageBox.information(self, "Email disabled",
                                        f"Archive saved to:\n{archive_path}\n\n"
                                        "Email is disabled or not configured in config.env.")
            return archive_path
        
        self.log_message(f"Emailing scan report to {self.email_notifier.recipient_email}...")
        self.send_email_async(f"Scan Report {scan_id}", self.email_notifier.send_scan_report,
                              scan_id, archive_path, summary, report_id=report_id)
        return archive_path

    def send_email_async(self, description: str, send_func, *args, report_id: Optional[int] = None, **kwargs):
        if not self.email_notifier.enabled:
            return
        thread = EmailThread(description, send_func, *args, **kwargs)
        thread.finished_with_status.connect(
            lambda ok, msg, rid=report_id: self.on_email_finished(ok, msg, rid))
        thread.finished.connect(lambda: self.email_threads.remove(thread) if thread in self.email_threads else None)
        self.email_threads.append(thread)
        thread.start()

    def on_email_finished(self, ok: bool, message: str, report_id: Optional[int]):
        self.log_message(message)
        if report_id is not None:
            self.db.update_report_status(report_id, 'sent' if ok else 'failed',
                                        None if ok else message)
            self.refresh_scan_history()

    def refresh_scan_history(self):
        """Reload the Scan History tab from the database."""
        try:
            scans = self.db.get_scan_overview()
        except Exception as e:
            self.log_message(f"Error reading scan history: {e}")
            return
        
        total_images = sum(s.get('image_count') or 0 for s in scans)
        total_logs = sum(s.get('detection_count') or 0 for s in scans)
        self.db_status_label.setText(
            f"Database: {self.db.db_path}   |   {len(scans)} scans   |   {total_logs} log rows   |   "
            f"{total_images} stored images   |   file size {self.db.get_database_size() / (1024 * 1024):.2f} MB")
        
        self.scan_table.setRowCount(0)
        for scan in scans:
            row = self.scan_table.rowCount()
            self.scan_table.insertRow(row)
            avg_temp = scan.get('avg_temperature_celsius')
            values = [
                scan.get('scan_id', ''),
                scan.get('start_time', ''),
                str(scan.get('max_weevil_count', 0)),
                f"{avg_temp:.1f}°C" if isinstance(avg_temp, (int, float)) else "N/A",
                str(scan.get('detection_count', 0)),
                f"{scan.get('image_count', 0)} ({(scan.get('image_total_bytes') or 0) / 1024:.0f} KB)",
                scan.get('report_status') or '-',
            ]
            for col, value in enumerate(values):
                self.scan_table.setItem(row, col, QTableWidgetItem(value))
        
        if self.scan_table.rowCount() and not self.scan_table.selectedItems():
            self.scan_table.selectRow(0)

    def selected_scan_id(self) -> Optional[str]:
        row = self.scan_table.currentRow()
        if row < 0:
            return None
        item = self.scan_table.item(row, 0)
        return item.text() if item else None

    def on_scan_selected(self):
        """Load the selected scan's stored detection log and images out of the database."""
        scan_id = self.selected_scan_id()
        self.image_list.clear()
        self.history_log_table.setRowCount(0)
        self.image_preview_label.setPixmap(QPixmap())
        if not scan_id:
            return
        
        try:
            detections = self.db.get_detections_by_scan(scan_id)
            images = self.db.get_scan_images(scan_id, include_blob=False)
        except Exception as e:
            self.log_message(f"Error loading scan {scan_id}: {e}")
            return
        
        for detection in detections:
            row = self.history_log_table.rowCount()
            self.history_log_table.insertRow(row)
            temp = detection.get('temperature_celsius')
            values = [
                detection.get('timestamp', ''),
                str(detection.get('weevil_count', 0)),
                f"{temp:.1f}°C" if isinstance(temp, (int, float)) else "N/A",
                detection.get('recommendation', ''),
            ]
            for col, value in enumerate(values):
                self.history_log_table.setItem(row, col, QTableWidgetItem(value))
        
        for image in images:
            item = QListWidgetItem(f"[{image['camera']}] {image['filename']} "
                                   f"({(image.get('image_bytes') or 0) / 1024:.0f} KB)")
            item.setData(Qt.UserRole, image['id'])
            self.image_list.addItem(item)
        
        self.image_preview_label.setText(
            f"{len(images)} image(s) stored for {scan_id}" if images
            else f"No images stored for {scan_id}")
        if images:
            self.image_list.setCurrentRow(0)

    def on_history_image_selected(self, row: int):
        """Render a stored image straight from its database BLOB."""
        if row < 0:
            return
        item = self.image_list.item(row)
        if item is None:
            return
        image_id = item.data(Qt.UserRole)
        scan_id = self.selected_scan_id()
        if not scan_id or image_id is None:
            return
        
        blob = next((i.get('image_blob') for i in self.db.get_scan_images(scan_id)
                     if i['id'] == image_id), None)
        if not blob:
            self.image_preview_label.setText("Image data missing from database")
            return
        
        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(blob), 'JPG'):
            self.image_preview_label.setText("Could not decode stored image")
            return
        self.image_preview_label.setPixmap(
            pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resend_selected_report(self):
        """Rebuild the selected scan's archive from the database and email it again."""
        scan_id = self.selected_scan_id()
        if not scan_id:
            QMessageBox.information(self, "No scan selected", "Select a scan in the table first.")
            return
        self.log_message(f"Rebuilding and emailing report for {scan_id} from the database...")
        self.email_scan_report(scan_id, interactive=True)
        self.refresh_scan_history()

    def export_selected_report(self):
        """Save the database-built archive for the selected scan to a chosen location."""
        scan_id = self.selected_scan_id()
        if not scan_id:
            QMessageBox.information(self, "No scan selected", "Select a scan in the table first.")
            return
        
        target, _ = QFileDialog.getSaveFileName(self, "Export scan report",
                                                os.path.join(self.reports_dir, f"{scan_id}_report.zip"),
                                                "Zip archives (*.zip)")
        if not target:
            return
        
        scan = self.db.get_scan_by_id(scan_id) or {}
        video_paths = [p for p in (scan.get('left_video_path'), scan.get('right_video_path')) if p]
        try:
            archive_path, info = report_builder.build_scan_archive(
                self.db, scan_id, target, video_paths=video_paths)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return
        
        if not archive_path:
            QMessageBox.warning(self, "Export failed", str(info.get('error')))
            return
        
        self.log_message(f"Exported {info['archive_name']} "
                         f"({info['archive_bytes'] / (1024 * 1024):.2f} MB) from the database")
        QMessageBox.information(self, "Export complete",
                                f"{info['log_entry_count']} log entries and {info['image_count']} images "
                                f"written to:\n{archive_path}")

    def view_previous_scans(self):
        # Open the previous scans folder in the system file explorer
        import subprocess
        import platform
        
        if os.path.exists(self.previous_scans_dir):
            system = platform.system()
            if system == "Windows":
                os.startfile(self.previous_scans_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", self.previous_scans_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.previous_scans_dir])
            self.log_message(f"Opened previous scans folder: {self.previous_scans_dir}")
        else:
            self.log_message("Previous scans folder does not exist yet")
    
    def save_scan_metadata(self) -> Optional[dict]:
        """Save scan metadata to database and JSON file, returning the summary."""
        import json
        
        if not self.current_scan_folder or not self.current_scan_id:
            return None
        
        # Calculate average temperature
        avg_temp = sum(self.scan_temp_readings) / len(self.scan_temp_readings) if self.scan_temp_readings else 0.0
        
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_count = self.db.get_scan_image_count(self.current_scan_id)
        duration = int((datetime.now() - self.scan_start_time).total_seconds()) if self.scan_start_time else None
        model_info = self.detector.get_model_info()
        metadata = {
            "scan_id": self.current_scan_id,
            "scan_start_time": self.scan_start_time.strftime("%Y-%m-%d %H:%M:%S") if self.scan_start_time else None,
            "scan_end_time": end_time_str,
            "duration_seconds": duration,
            "protocol_duration_seconds": self.scan_duration_seconds,
            "max_weevil_count": self.scan_max_count,
            "average_temperature_celsius": round(avg_temp, 2) if avg_temp is not None else None,
            "temperature_readings_count": len(self.scan_temp_readings),
            "log_entry_count": self.scan_detection_count,
            "image_count": image_count,
            "recommendation": self.logger.generate_recommendation(self.scan_max_count, self.is_after_mixing),
            "after_mixing": self.is_after_mixing,
            "model": {
                "name": model_info['model_name'],
                "architecture": model_info['architecture'],
                "backend": model_info['backend'],
                "imgsz": model_info['imgsz'],
                "confidence_threshold": model_info['confidence_threshold'],
                "classes": model_info['classes'],
            },
            "avg_inference_ms": round(self.detector.avg_inference_ms, 1),
            "videos": {
                "left_camera": "left_camera.mp4",
                "right_camera": "right_camera.mp4"
            }
        }
        
        # Save to database (metadata_json makes the scan row self-describing)
        self.db.update_scan(
            self.current_scan_id,
            end_time_str,
            self.scan_max_count,
            round(avg_temp, 2) if avg_temp is not None else None,
            len(self.scan_temp_readings),
            json.dumps(metadata)
        )
        
        # Also drop a copy next to the recordings for offline inspection
        metadata_path = os.path.join(self.current_scan_folder, "scan_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.log_message(f"Scan metadata saved to database ({self.scan_detection_count} log entries, "
                         f"{image_count} images stored)")
        return metadata

    def set_red_light(self):
        self.led_controller.set_red()
        self.log_message("Red light activated")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "Red light activated to lure rice weevils")

    def set_white_light(self):
        self.led_controller.set_white()
        self.log_message("White light activated")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "White light activated for detection")

    def toggle_mixing_state(self):
        self.is_after_mixing = not self.is_after_mixing
        if self.is_after_mixing:
            self.mixing_button.setText("After Mixing: ON")
            self.mixing_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 13px;
                    border-radius: 5px;
                    padding: 6px;
                }
            """)
            self.log_message("Marked as after mixing/sifting")
        else:
            self.mixing_button.setText("Mark as After Mixing/Sifting")
            self.mixing_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-size: 13px;
                    border-radius: 5px;
                    padding: 6px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            self.log_message("Marked as before mixing/sifting")

    def set_leds_off(self):
        self.led_controller.off()
        self.log_message("LEDs turned off")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "LEDs turned off")

    def update_left_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        self.latest_annotated_left = annotated_frame
        
        # Write to video file if recording
        if self.video_writer_left and self.video_writer_left.isOpened():
            self.video_writer_left.write(annotated_frame)
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.left_camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.left_camera_label.setPixmap(scaled_pixmap)

    def update_right_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        self.latest_annotated_right = annotated_frame
        
        # Write to video file if recording
        if self.video_writer_right and self.video_writer_right.isOpened():
            self.video_writer_right.write(annotated_frame)
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.right_camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.right_camera_label.setPixmap(scaled_pixmap)

    def update_detection(self, count: int, confidence: float, activity: str):
        # Update count label (always update real-time)
        self.count_label.setText(f"Weevil Count: {count}")
        self.confidence_label.setText(
            f"Avg Confidence: {confidence * 100:.1f}%" if confidence else "Avg Confidence: --")
        
        # Get temperature
        temperature = self.temp_sensor.read_temperature()
        
        # Track scan metadata
        if self.is_scanning:
            if count > self.scan_max_count:
                self.scan_max_count = count
            if temperature is not None:
                self.scan_temp_readings.append(temperature)
        
        current_time = datetime.now()
        # "Significant" means a noticeably high reading, not merely a non-zero one.
        is_significant = count >= self.high_weevil_threshold
        
        # Images are only captured for significant counts, rate-limited by the cooldown.
        should_capture = (self.is_scanning and is_significant and
                         (self.last_image_capture_time is None or
                          (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_cooldown))
        
        # Log every significant reading (rate-limited so a sustained high count does not
        # flood the table), plus a sparse baseline row so quiet scans still record that
        # the equipment was running and found little or nothing.
        should_log_significant = is_significant and (
            self.last_significant_log_time is None or
            (current_time - self.last_significant_log_time).total_seconds() >= self.significant_log_interval_seconds)
        should_log_baseline = (self.last_log_time is None or
                               current_time - self.last_log_time >= timedelta(seconds=self.log_interval_seconds))
        should_log = should_log_significant or should_log_baseline
        
        # Always log to the CSV file (for data integrity)
        log_entry = self.logger.log_detection(count, temperature, self.is_after_mixing, activity)
        
        # Update recommendation (always update real-time)
        self.recommendation_label.setText(f"Recommendation: {log_entry.recommendation}")
        
        if not (should_log or should_capture):
            return
        
        # Persist the detection row, then attach any captured images to it
        detection_id = None
        if self.is_scanning and self.current_scan_id:
            detection_id = self.db.add_detection(
                self.current_scan_id,
                log_entry.timestamp,
                count,
                temperature,
                log_entry.recommendation,
                activity
            )
            self.scan_detection_count += 1
        
        if should_capture:
            self.capture_detection_images(count, confidence, detection_id)
            self.last_image_capture_time = current_time
        
        if should_log:
            temp_str = f"{temperature:.1f}°C" if temperature is not None else "N/A"
            marker = "HIGH COUNT " if should_log_significant else ""
            self.log_message(f"{marker}{log_entry.timestamp} - Count: {count}, Temp: {temp_str}, "
                             f"Rec: {log_entry.recommendation}")
            self.add_detection_to_table(log_entry.timestamp, count, temperature, log_entry.recommendation)
            self.last_log_time = current_time
            if should_log_significant:
                self.last_significant_log_time = current_time
    
    def capture_detection_images(self, count: int, confidence: float = 0.0,
                                detection_id: Optional[int] = None) -> list:
        """Store annotated snapshots of the detected rice weevils in the database (and on disk).
        
        The JPEG bytes go into scan_images so the emailing system can rebuild the report
        archive from the database alone.
        """
        if not self.current_scan_id:
            return []
        
        timestamp = datetime.now()
        stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        # Images are only captured for significant counts, so they are all high-count frames.
        prefix = "high_weevil"
        quality = int(os.getenv('IMAGE_JPEG_QUALITY', '85'))
        saved = []
        
        for side, frame in (("left", self.latest_annotated_left), ("right", self.latest_annotated_right)):
            if frame is None:
                continue
            filename = f"{prefix}_{side}_{stamp}_count{count}.jpg"
            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                continue
            
            # Keep a copy on disk next to the recordings for offline browsing
            path = os.path.join(self.scan_images_dir, filename) if self.scan_images_dir else None
            if path:
                try:
                    with open(path, 'wb') as f:
                        f.write(buffer.tobytes())
                except OSError as e:
                    self.log_message(f"Could not write image to disk ({e}); database copy still stored")
                    path = None
            
            try:
                self.db.add_scan_image(
                    scan_id=self.current_scan_id,
                    timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    camera=side,
                    filename=filename,
                    image_blob=buffer.tobytes(),
                    weevil_count=count,
                    confidence_avg=round(confidence, 4) if confidence else None,
                    file_path=path,
                    detection_id=detection_id,
                )
                saved.append(filename)
                self.scan_image_count += 1
            except Exception as e:
                self.log_message(f"Error storing image in database: {e}")
        
        if saved:
            self.log_message(f"High weevil count ({count} >= {self.high_weevil_threshold}) - "
                             f"stored {len(saved)} image(s) in database")
        return saved

    def add_detection_to_table(self, timestamp: str, count: int, temperature: Optional[float], recommendation: str):
        temp_str = f"{temperature:.1f}°C" if temperature is not None else "N/A"
        
        row_position = self.detection_table.rowCount()
        self.detection_table.insertRow(row_position)
        
        self.detection_table.setItem(row_position, 0, QTableWidgetItem(timestamp))
        self.detection_table.setItem(row_position, 1, QTableWidgetItem(str(count)))
        self.detection_table.setItem(row_position, 2, QTableWidgetItem(temp_str))
        self.detection_table.setItem(row_position, 3, QTableWidgetItem(recommendation))
        
        # Auto-scroll to bottom
        self.detection_table.scrollToBottom()
        
        # Limit table to last 100 entries
        if self.detection_table.rowCount() > 100:
            self.detection_table.removeRow(0)

    def update_temperature(self):
        temperature = self.temp_sensor.read_temperature()
        if temperature is not None:
            self.temp_label.setText(f"Temperature: {temperature:.1f}°C")
    
    def update_clock(self):
        from datetime import datetime
        now = datetime.now()
        # Format: June 25, 2026 9:31:45 PM
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        
        # Update date and time labels on Live Feed tab
        self.date_label.setText(f"Date: {date_str}")
        self.time_label.setText(f"Time: {time_str}")

    def log_message(self, message: str):
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        if self.is_scanning:
            self.stop_scan()
        # Let any in-flight report email finish so the scan report is not lost on exit
        for thread in list(self.email_threads):
            thread.wait(90000)
        self.led_controller.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Show loading screen first
    logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
    from src.gui.loading_screen import LoadingScreen
    loading_screen = LoadingScreen(logo_path)
    loading_screen.show()
    
    # Create main window but don't show it yet
    window = MainWindow()
    
    # Connect loading screen completion to show main window
    loading_screen.loading_complete.connect(lambda: show_main_window(loading_screen, window))
    
    sys.exit(app.exec_())


def show_main_window(loading_screen, main_window):
    """Transition from loading screen to main window"""
    loading_screen.close()
    main_window.show()


if __name__ == '__main__':
    main()
