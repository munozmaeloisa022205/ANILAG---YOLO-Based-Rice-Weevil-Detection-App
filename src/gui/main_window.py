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


class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str)

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
                "Detection"
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
        
        self.detection_thread: Optional[DetectionThread] = None
        self.is_scanning = False
        self.is_after_mixing = False
        
        # Detection data for table
        self.detection_data = []  # List of tuples: (timestamp, count, temperature, recommendation)
        
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
            logo_label.setStyleSheet("color: #0066cc;")
        header_layout.addWidget(logo_label)
        
        # Title label
        title_label = QLabel("Rice Weevil Detection System")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #333;")
        header_layout.addWidget(title_label)
        
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
        self.live_feed_tab.setLayout(layout)
        
        # Left panel - Camera feeds (large) and controls below
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=3)
        
        # Camera feeds container - larger
        camera_container = QHBoxLayout()
        left_panel.addLayout(camera_container, stretch=3)
        
        # Left camera feed label - larger
        self.left_camera_label = QLabel()
        self.left_camera_label.setMinimumSize(640, 480)
        self.left_camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.left_camera_label.setAlignment(Qt.AlignCenter)
        self.left_camera_label.setText("Left Camera")
        camera_container.addWidget(self.left_camera_label)
        
        # Right camera feed label - larger
        self.right_camera_label = QLabel()
        self.right_camera_label.setMinimumSize(640, 480)
        self.right_camera_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.right_camera_label.setAlignment(Qt.AlignCenter)
        self.right_camera_label.setText("Right Camera")
        camera_container.addWidget(self.right_camera_label)
        
        # Scan controls below camera feeds
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
        left_panel.addWidget(scan_group)
        
        # LED controls below scan controls
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
        left_panel.addWidget(led_group)
        
        # Right panel - Compact detection log
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=1)
        
        # Detection log display - compact
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

    def setup_detection_info_tab(self):
        layout = QVBoxLayout()
        self.detection_info_tab.setLayout(layout)
        
        # Real-time clock display
        clock_label = QLabel()
        clock_label.setAlignment(Qt.AlignCenter)
        clock_label.setFont(QFont("Arial", 16, QFont.Bold))
        clock_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.clock_label = clock_label
        layout.addWidget(clock_label)
        
        # Current detection info
        current_info_group = QGroupBox("Current Detection")
        current_info_layout = QGridLayout()
        
        self.count_label = QLabel("Weevil Count: 0")
        self.count_label.setFont(QFont("Arial", 14, QFont.Bold))
        current_info_layout.addWidget(self.count_label, 0, 0)
        
        self.temp_label = QLabel("Temperature: --°C")
        self.temp_label.setFont(QFont("Arial", 12))
        current_info_layout.addWidget(self.temp_label, 0, 1)
        
        self.recommendation_label = QLabel("Recommendation: --")
        self.recommendation_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.recommendation_label.setStyleSheet("color: #0066cc;")
        current_info_layout.addWidget(self.recommendation_label, 1, 0, 1, 2)
        
        current_info_group.setLayout(current_info_layout)
        layout.addWidget(current_info_group)
        
        # Detection history table
        table_group = QGroupBox("Detection History")
        table_layout = QVBoxLayout()
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(["Timestamp", "Count", "Temperature", "Recommendation"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detection_table.setEditTriggers(QTableWidget.NoEditTriggers)
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
        
        self.detection_thread = DetectionThread(self.camera_manager, self.detector)
        self.detection_thread.frame_ready_left.connect(self.update_left_frame)
        self.detection_thread.frame_ready_right.connect(self.update_right_frame)
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
        
        self.camera_manager.stop()
        
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

    def update_left_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        
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
        # Update count label
        self.count_label.setText(f"Weevil Count: {count}")
        
        # Get temperature
        temperature = self.temp_sensor.read_temperature()
        
        # Log detection
        log_entry = self.logger.log_detection(count, temperature, self.is_after_mixing, activity)
        
        # Update recommendation
        self.recommendation_label.setText(f"Recommendation: {log_entry.recommendation}")
        
        # Update log display
        temp_str = f"{temperature:.1f}°C" if temperature is not None else "N/A"
        self.log_message(f"{log_entry.timestamp} - Count: {count}, Temp: {temp_str}, Rec: {log_entry.recommendation}")
        
        # Add to detection table
        self.add_detection_to_table(log_entry.timestamp, count, temperature, log_entry.recommendation)
        
        # Send email notification
        if count > 0 or self.is_after_mixing:
            self.email_notifier.send_detection_alert(
                log_entry.timestamp,
                count,
                temperature,
                log_entry.recommendation,
                activity
            )

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
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        self.clock_label.setText(f"{date_str}  {time_str}")

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
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
