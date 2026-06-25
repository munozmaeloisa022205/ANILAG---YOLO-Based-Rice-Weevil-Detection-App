import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QTextEdit, QGridLayout, QGroupBox,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from typing import Optional
import os
from dotenv import load_dotenv

# Import modules
from src.hardware.camera import Camera, DualCameraManager
from src.hardware.temperature import TemperatureSensor
from src.hardware.led_controller import LEDController
from src.detection.yolov11_detector import YOLOv11Detector, DetectionResult
from src.logging.logger import DetectionLogger
from src.notification.email_notifier import EmailNotifier
from src.backend.database import get_database


class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str, np.ndarray, np.ndarray)  # Added frames for image capture

    def __init__(self, camera_manager: DualCameraManager, detector: YOLOv11Detector):
        super().__init__()
        self.camera_manager = camera_manager
        self.detector = detector
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            left_frame = self.camera_manager.get_left_frame()
            right_frame = self.camera_manager.get_right_frame()
            
            if left_frame is not None:
                detection_left = self.detector.detect(left_frame)
                self.frame_ready_left.emit(left_frame, detection_left)
            
            if right_frame is not None:
                detection_right = self.detector.detect(right_frame)
                self.frame_ready_right.emit(right_frame, detection_right)
            
            # Emit combined detection update
            total_count = 0
            if left_frame is not None:
                detection_left = self.detector.detect(left_frame)
                total_count += detection_left.count
            if right_frame is not None:
                detection_right = self.detector.detect(right_frame)
                total_count += detection_right.count
            
            self.detection_update.emit(
                total_count,
                0.0,  # Placeholder for confidence
                "Detection",
                left_frame if left_frame is not None else None,
                right_frame if right_frame is not None else None
            )
            self.msleep(50)  # 20 FPS

    def stop(self):
        self.running = False
        self.wait()


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
            fps=int(os.getenv('CAMERA_FPS', '30')),
            camera_type=os.getenv('CAMERA_TYPE', 'opencv')
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
        
        # Initialize database
        self.db = get_database(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'anilag.db'))
        
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
        self.previous_scans_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'previous_scans')
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

    def initialize_components(self):
        # Initialize detector
        if self.detector.initialize():
            self.log_message("YOLOv11 detector initialized successfully")
        else:
            self.log_message("Failed to initialize YOLOv11 detector")
        
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
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_scan_folder = os.path.join(self.previous_scans_dir, f"scan_{timestamp}")
        os.makedirs(self.current_scan_folder, exist_ok=True)
        
        # Generate scan ID for database
        self.current_scan_id = f"scan_{timestamp}"
        
        # Initialize video writers with compression
        width = int(os.getenv('CAMERA_WIDTH', '640'))
        height = int(os.getenv('CAMERA_HEIGHT', '480'))
        fps = int(os.getenv('CAMERA_FPS', '30'))
        
        # Use H.264 codec for better compression
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        if fourcc == -1:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback
        
        left_video_path = os.path.join(self.current_scan_folder, "left_camera.mp4")
        right_video_path = os.path.join(self.current_scan_folder, "right_camera.mp4")
        
        self.video_writer_left = cv2.VideoWriter(left_video_path, fourcc, fps, (width, height))
        self.video_writer_right = cv2.VideoWriter(right_video_path, fourcc, fps, (width, height))
        
        # Reset scan metadata
        self.scan_start_time = datetime.now()
        self.scan_max_count = 0
        self.scan_avg_temp = 0.0
        self.scan_temp_readings = []
        
        # Create scan record in database
        start_time_str = self.scan_start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.db.create_scan(self.current_scan_id, start_time_str, left_video_path, right_video_path)
        
        self.detection_thread = DetectionThread(self.camera_manager, self.detector)
        self.detection_thread.frame_ready_left.connect(self.update_left_frame)
        self.detection_thread.frame_ready_right.connect(self.update_right_frame)
        self.detection_thread.detection_update.connect(self.update_detection)
        self.detection_thread.start()
        
        # Reset image capture tracking
        self.last_image_capture_time = None
        
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
        self.log_message(f"Scan started - Recording to {self.current_scan_folder}")
        self.email_notifier.send_activity_log("Scan Started", "Live detection has been initiated")

    def stop_scan(self):
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
        
        # Save scan metadata to database and file
        if self.current_scan_folder and self.current_scan_id:
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
        self.email_notifier.send_activity_log("Scan Stopped", "Live detection has been stopped")

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
    
    def save_scan_metadata(self):
        """Save scan metadata to database and JSON file"""
        import json
        from datetime import datetime
        
        if not self.current_scan_folder or not self.current_scan_id:
            return
        
        # Calculate average temperature
        avg_temp = sum(self.scan_temp_readings) / len(self.scan_temp_readings) if self.scan_temp_readings else 0.0
        
        # Save to database
        end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.update_scan(
            self.current_scan_id,
            end_time_str,
            self.scan_max_count,
            round(avg_temp, 2) if avg_temp else None,
            len(self.scan_temp_readings)
        )
        
        # Save to JSON file
        metadata = {
            "scan_start_time": self.scan_start_time.strftime("%Y-%m-%d %H:%M:%S") if self.scan_start_time else None,
            "scan_end_time": end_time_str,
            "max_weevil_count": self.scan_max_count,
            "average_temperature_celsius": round(avg_temp, 2) if avg_temp else None,
            "temperature_readings_count": len(self.scan_temp_readings),
            "videos": {
                "left_camera": "left_camera.mp4",
                "right_camera": "right_camera.mp4"
            }
        }
        
        metadata_path = os.path.join(self.current_scan_folder, "scan_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.log_message(f"Scan metadata saved to database and {metadata_path}")

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

    def update_left_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        
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

    def update_detection(self, count: int, confidence: float, activity: str, left_frame: np.ndarray = None, right_frame: np.ndarray = None):
        from datetime import datetime, timedelta
        
        # Update count label (always update real-time)
        self.count_label.setText(f"Weevil Count: {count}")
        
        # Get temperature
        temperature = self.temp_sensor.read_temperature()
        
        # Track scan metadata
        if self.is_scanning:
            if count > self.scan_max_count:
                self.scan_max_count = count
            if temperature is not None:
                self.scan_temp_readings.append(temperature)
            
            # Capture image if weevil count is high and cooldown has passed
            if count >= self.high_weevil_threshold:
                current_time = datetime.now()
                if self.last_image_capture_time is None or \
                   (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_cooldown:
                    self.capture_high_weevil_image(left_frame, right_frame, count, temperature)
                    self.last_image_capture_time = current_time
        
        # Check if 1 minute has passed since last log
        current_time = datetime.now()
        should_log = False
        
        if self.last_log_time is None:
            should_log = True
        elif current_time - self.last_log_time >= timedelta(minutes=1):
            should_log = True
        
        # Always log to file (for data integrity), but only update UI log display and table every 1 minute
        log_entry = self.logger.log_detection(count, temperature, self.is_after_mixing, activity)
        
        # Save to database if scanning
        if self.is_scanning and self.current_scan_id and should_log:
            self.db.add_detection(
                self.current_scan_id,
                log_entry.timestamp,
                count,
                temperature,
                log_entry.recommendation,
                activity
            )
        
        # Update recommendation (always update real-time)
        self.recommendation_label.setText(f"Recommendation: {log_entry.recommendation}")
        
        # Only update log display and table every 1 minute
        if should_log:
            temp_str = f"{temperature:.1f}°C" if temperature is not None else "N/A"
            self.log_message(f"{log_entry.timestamp} - Count: {count}, Temp: {temp_str}, Rec: {log_entry.recommendation}")
            
            # Add to detection table
            self.add_detection_to_table(log_entry.timestamp, count, temperature, log_entry.recommendation)
            
            # Update last log time
            self.last_log_time = current_time
    
    def capture_high_weevil_image(self, left_frame: np.ndarray, right_frame: np.ndarray, count: int, temperature: float):
        """Capture and save images when weevil count is high"""
        if not self.current_scan_folder:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save left camera image
        if left_frame is not None:
            left_image_path = os.path.join(self.current_scan_folder, f"high_weevil_left_{timestamp}.jpg")
            cv2.imwrite(left_image_path, left_frame)
            self.log_message(f"High weevil count ({count}) - Saved left image: {left_image_path}")
        
        # Save right camera image
        if right_frame is not None:
            right_image_path = os.path.join(self.current_scan_folder, f"high_weevil_right_{timestamp}.jpg")
            cv2.imwrite(right_image_path, right_frame)
            self.log_message(f"High weevil count ({count}) - Saved right image: {right_image_path}")

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
