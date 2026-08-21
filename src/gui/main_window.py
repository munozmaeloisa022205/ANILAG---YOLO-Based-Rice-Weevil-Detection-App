import sys
import cv2
import numpy as np

# IMPORTANT: the detector (and therefore torch) must be imported BEFORE PyQt5.
# On Windows, loading Qt first makes torch's DLL load fail with
#   OSError: [WinError 1114] ... Error loading "...\torch\lib\c10.dll"
# because Qt has already pulled in a conflicting runtime. Importing torch first
# is harmless everywhere else, including the Raspberry Pi. Do not reorder.
from src.detection.yolov11_detector import YOLOv11Detector, DetectionResult

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QGroupBox,
                             QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
                             QListWidget, QListWidgetItem, QSplitter, QSizePolicy,
                             QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from typing import Optional, TextIO
import io
import os
import shutil
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import modules
from src.hardware.camera import DualCameraManager
from src.hardware.temperature import TemperatureSensor
from src.hardware.led_controller import LEDController
from src.logging.logger import DetectionLogger
from src.notification.email_notifier import EmailNotifier
from src.backend.database import get_database
from src.backend import report_builder


class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    detection_update = pyqtSignal(int, float, str)
    stats_update = pyqtSignal(float, float)  # detections per second, avg inference ms
    camera_status = pyqtSignal(bool, bool)  # left has live signal, right has live signal

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

            # TEMPORARY TEST MODE: bypass is_healthy() so inference runs even
            # when the signal validation reports blank/frozen frames.
            left_healthy = self.camera_manager.left_camera.is_running()
            right_healthy = self.camera_manager.right_camera.is_running()
            self.camera_status.emit(left_healthy, right_healthy)
            left_frame = self.camera_manager.get_left_frame() if left_healthy else None
            right_frame = self.camera_manager.get_right_frame() if right_healthy else None

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


class PreviewThread(QThread):
    """Live camera preview without detection overhead.

    Runs continuously from app launch so both camera feeds are visible before
    the user clicks Start Scan. It only reads and emits frames - no YOLO
    inference - so the Pi 5 CPU stays free for the GUI. When a scan starts the
    preview thread is swapped out for the DetectionThread, and when the scan
    stops the preview thread is started again so the feed never goes dark.
    """
    frame_ready_left = pyqtSignal(np.ndarray)
    frame_ready_right = pyqtSignal(np.ndarray)
    camera_status = pyqtSignal(bool, bool)

    def __init__(self, camera_manager: DualCameraManager, interval_ms: int = 66):
        super().__init__()
        self.camera_manager = camera_manager
        # ~15 FPS is plenty for a live preview and keeps the Pi 5 cool.
        self.interval_ms = max(0, interval_ms)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            cycle_started = time.perf_counter()
            left_healthy = self.camera_manager.left_camera.is_running()
            right_healthy = self.camera_manager.right_camera.is_running()
            self.camera_status.emit(left_healthy, right_healthy)
            left_frame = self.camera_manager.get_left_frame() if left_healthy else None
            right_frame = self.camera_manager.get_right_frame() if right_healthy else None
            if left_frame is not None:
                self.frame_ready_left.emit(left_frame)
            if right_frame is not None:
                self.frame_ready_right.emit(right_frame)
            elapsed_ms = (time.perf_counter() - cycle_started) * 1000
            remaining = self.interval_ms - elapsed_ms
            self.msleep(int(remaining) if remaining > 0 else 1)

    def stop(self):
        self.running = False
        self.wait()


class ConsoleLogStream(QObject):
    """Tee stdout/stderr so every print() also appears in the System Log.

    print() is called from background threads (camera capture loop, email
    thread, etc.), so a Qt signal is used to marshal the text to the GUI
    thread safely.
    """
    line_printed = pyqtSignal(str)

    def __init__(self, original: TextIO):
        super().__init__()
        self._original = original
        self._buffer = ""

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.line_printed.emit(line)
        return len(text)

    def flush(self):
        self._original.flush()
        if self._buffer.strip():
            self.line_printed.emit(self._buffer)
            self._buffer = ""


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


class CameraLabel(QLabel):
    """A QLabel that keeps its zoom control panel pinned to the right edge
    inside the frame when the label (and its pixmap) is resized.

    Supports click-and-drag panning when the feed is zoomed in, so the user
    can navigate to any part of the zoomed frame without the layout shifting.
    """
    ZOOM_PANEL_MARGIN = 10

    # Emitted when the user drags the feed to pan a zoomed view.
    # Arguments: delta_x, delta_y in frame-pixel units (already scaled by zoom).
    pan_requested = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_panel = None
        self._dragging = False
        self._drag_start = None

    def position_zoom_panel(self):
        """Pin the panel to the right edge, vertically centered, fully inside the frame."""
        panel = self.zoom_panel
        if panel is None:
            return
        x = self.width() - panel.width() - self.ZOOM_PANEL_MARGIN
        y = (self.height() - panel.height()) // 2
        panel.move(max(self.ZOOM_PANEL_MARGIN, x), max(self.ZOOM_PANEL_MARGIN, y))
        panel.raise_()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None:
            delta = event.pos() - self._drag_start
            self._drag_start = event.pos()
            # Emit the pan delta; the main window converts screen pixels to
            # frame-pixel offsets based on the current zoom factor.
            self.pan_requested.emit(float(delta.x()), float(delta.y()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self._drag_start = None
            self.setCursor(Qt.OpenHandCursor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_zoom_panel()

    def showEvent(self, event):
        # The hidden tab never gets a resize until it is first shown, so place the
        # panel here too - otherwise it sits in the top-left corner over the title.
        super().showEvent(event)
        self.position_zoom_panel()


class MainWindow(QMainWindow):
    # Touch target sizing for the 7-inch (1024x600) Raspberry Pi touchscreen.
    # 64px is roughly a 9mm square on that panel - comfortably finger-sized.
    TOUCH_BUTTON_SIZE = 52
    # Overlay zoom buttons are smaller than the main controls but still touchable.
    ZOOM_BUTTON_SIZE = 40
    # Fixed width of the compact info column (Current Detection / Detection Model
    # / System Log). Kept narrow so the camera feed keeps the dominant share of
    # the 1024px-wide screen, but wide enough for the readout text to be legible.
    INFO_PANEL_WIDTH = 240
    # Table/list rows in the Detection Logs tab are selected by finger, so they need
    # to be at least this tall to be reliably tappable on the 7-inch panel.
    TOUCH_ROW_HEIGHT = 44
    # Scrollbars are dragged by finger, so they are far wider than the desktop default.
    TOUCH_SCROLLBAR_SIZE = 22
    # Group boxes in the feed tabs: a short title margin keeps the controls strip
    # and the info column from eating vertical space on the 600px-tall screen.
    COMPACT_GROUP_STYLE = (
        "QGroupBox { font-weight: bold; font-size: 11px; border: 1px solid #ccc; "
        "border-radius: 5px; margin-top: 7px; padding-top: 2px; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anilag - Rice Weevil Detection and Control System")
        # Target display is the Raspberry Pi 5 touchscreen at 1024x600.
        # Start maximized so the camera feeds use all available screen space.
        self.setGeometry(50, 50, 1024, 600)
        self.setMinimumSize(800, 480)
        self.showMaximized()

        # Tee stdout/stderr into the System Log so every print() from any
        # module (camera, detector, email, LED, temperature) appears in the
        # GUI just like it does in the terminal.
        self._console_stream = ConsoleLogStream(sys.stdout)
        self._console_stream_err = ConsoleLogStream(sys.stderr)
        sys.stdout = self._console_stream
        sys.stderr = self._console_stream_err
        
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
        self.preview_thread: Optional[PreviewThread] = None
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

        # Per-camera digital zoom (applied to the displayed feed only; the recorded
        # video and stored images keep the full frame). Zoom factor of 1.0 = no zoom,
        # 2.0 = 2x center crop, etc. Capped to keep the Pi 5 crop/resize cheap.
        self.left_zoom = 1.0
        self.right_zoom = 1.0
        # Pan offsets (in original-frame pixels) so the user can drag the zoomed
        # view to navigate to any part of the frame. Reset to centre on zoom change.
        self.left_pan_x = 0.0
        self.left_pan_y = 0.0
        self.right_pan_x = 0.0
        self.right_pan_y = 0.0
        self.zoom_step = 0.5
        self.zoom_min = 1.0
        self.zoom_max = 5.0
        
        # Scan protocol: each scan runs for a fixed duration, then auto-stops and emails the report
        self.scan_duration_seconds = int(os.getenv('SCAN_DURATION_SECONDS', '180'))
        self.scan_images_dir = None
        self.email_threads = []
        # Sparse baseline log interval - guarantees a scan always has log rows, even when
        # nothing significant is found, so "no weevils" is recorded rather than missing.
        self.log_interval_seconds = int(os.getenv('LOG_INTERVAL_SECONDS', '60'))
        # Minimum gap between log rows triggered by a frame containing weevils
        self.detection_log_interval_seconds = int(os.getenv(
            'DETECTION_LOG_INTERVAL_SECONDS',
            os.getenv('SIGNIFICANT_LOG_INTERVAL_SECONDS', '5')))
        self.last_detection_log_time = None
        # Tracks which DS18B20 conversion has already been added to the scan average
        self._last_temp_sample_id = None
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
        
        # Paint both immediately so the UI never shows a blank clock or "--" temperature
        # for the first second after startup.
        self.update_clock()
        self.update_temperature()
        
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
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(8)
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #f5f5f5; border-bottom: 2px solid #ddd;")
        header_widget.setFixedHeight(64)

        # Logo label - fills the header height, no box/border around it
        self.logo_label = QLabel()
        self.logo_label.setStyleSheet("background: transparent; border: none; padding: 0px;")
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        self._logo_path = logo_path
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale to fill the header height (60px fits the 64px header with 2px margin)
            pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setMinimumSize(60, 60)
        else:
            self.logo_label.setText("ANILAG")
            self.logo_label.setFont(QFont("Arial", 14, QFont.Bold))
            self.logo_label.setStyleSheet("color: #2E7D32; background: transparent; border: none;")
        header_layout.addWidget(self.logo_label)

        # Title and tagline container - vertically centered next to the logo
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_container.setLayout(title_layout)

        # Title label
        title_label = QLabel("Anilag")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2E7D32; background: transparent;")
        title_layout.addWidget(title_label)

        # Tagline label
        tagline_label = QLabel("Rice Weevil Detection and Control System")
        tagline_label.setFont(QFont("Arial", 9))
        tagline_label.setStyleSheet("color: #666; background: transparent;")
        title_layout.addWidget(tagline_label)

        header_layout.addWidget(title_container)
        
        header_layout.addStretch()
        
        # Always-visible real-time clock (updated every second from the system clock)
        self.header_clock_label = QLabel()
        self.header_clock_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.header_clock_label.setStyleSheet("color: #2E7D32; padding-right: 6px;")
        self.header_clock_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.header_clock_label)
        
        main_layout.addWidget(header_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Camera feed tabs. Both tabs share one instance of the Scan Controls,
        # LED Controls, Current Detection and Detection Log panels, which are
        # re-parented into whichever tab is active. Only the camera feed differs.
        self.build_shared_panels()
        self.left_feed_tab = self.create_feed_tab('left')
        self.tab_widget.addTab(self.left_feed_tab, "Left Camera Feed")
        self.right_feed_tab = self.create_feed_tab('right')
        self.tab_widget.addTab(self.right_feed_tab, "Right Camera Feed")
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.attach_shared_panels('left')
        
        # Detection Logs tab - merges the old real-time Detection Information
        # table with the database-backed scan history. The per-scan detection log
        # and captured images are read back from SQLite, so a separate live table
        # would only duplicate what the shared System Log panel already shows.
        self.scan_history_tab = QWidget()
        self.setup_scan_history_tab()
        self.tab_widget.addTab(self.scan_history_tab, "Detection Logs")
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # Route stdout/stderr print output into the System Log so the GUI
        # shows the same text that appears in the terminal.
        self._console_stream.line_printed.connect(self._on_console_line)
        self._console_stream_err.line_printed.connect(self._on_console_line)

    def _touch_button_style(self, background, hover, pressed=None, color="white", border="none"):
        return f"""
            QPushButton {{
                background-color: {background};
                color: {color};
                font-size: 13px;
                font-weight: bold;
                border: {border};
                border-radius: 6px;
                padding: 6px;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:pressed {{ background-color: {pressed or hover}; }}
        """

    def _make_touch_button(self, text, background, hover, handler,
                           pressed=None, color="white", border="none"):
        """Finger-sized button for the touchscreen. Height stays touch-safe while
        the width shrinks to fit the narrower controls strip under the feed."""
        btn = QPushButton(text)
        btn.setFixedHeight(self.TOUCH_BUTTON_SIZE)
        btn.setMinimumWidth(64)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setStyleSheet(self._touch_button_style(background, hover, pressed, color, border))
        btn.clicked.connect(handler)
        return btn

    def _make_zoom_button(self, text, handler):
        """Small overlay button used by the per-camera digital zoom controls."""
        btn = QPushButton(text)
        btn.setFixedSize(self.ZOOM_BUTTON_SIZE, self.ZOOM_BUTTON_SIZE)
        btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(69, 90, 100, 220); color: white;
                font-size: 20px; font-weight: bold;
                border-radius: 6px; padding: 1px;
            }
            QPushButton:hover { background-color: rgba(96, 125, 139, 240); }
            QPushButton:pressed { background-color: rgba(55, 71, 79, 255); }
        """)
        btn.clicked.connect(handler)
        return btn

    def _make_touch_scrollable(self, widget):
        """Make a QTableWidget/QListWidget scroll comfortably on the 7-inch
        touchscreen by widening its scrollbars and giving the handles a tall,
        finger-friendly minimum size. The styling is applied to the widget's
        own viewport scrollbars so it composes cleanly with any per-widget
        stylesheet already in place."""
        size = self.TOUCH_SCROLLBAR_SIZE
        style = f"""
            QScrollBar:vertical {{
                background: #f0f0f0;
                width: {size}px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #9e9e9e;
                border-radius: {size // 2}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:pressed {{ background: #616161; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
            QScrollBar:horizontal {{
                background: #f0f0f0;
                height: {size}px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: #9e9e9e;
                border-radius: {size // 2}px;
                min-width: 40px;
            }}
            QScrollBar::handle:horizontal:pressed {{ background: #616161; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}
        """
        # Merge with any stylesheet already set on the widget so we don't
        # clobber table/list styling applied by the caller. Bare properties
        # (no selector) must be wrapped in a selector block, otherwise Qt
        # cannot parse the concatenation of bare properties + selector rules.
        existing = widget.styleSheet().strip()
        if existing and '{' not in existing:
            existing = f"QWidget {{ {existing} }}"
        widget.setStyleSheet(existing + style)

    def _build_zoom_overlay(self, parent_label, side):
        """Create a vertical zoom control panel overlaid inside a camera label
        (+ on top, zoom label, - on bottom) pinned to the right edge."""
        panel = QWidget(parent_label)
        panel.setObjectName("zoomPanel")
        panel.setStyleSheet("#zoomPanel { background-color: rgba(0, 0, 0, 110); border-radius: 8px; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(4)
        zoom_in = self._make_zoom_button("+", lambda: self.adjust_zoom(side, 1))
        zoom_out = self._make_zoom_button("-", lambda: self.adjust_zoom(side, -1))
        zoom_label = QLabel("1.0x")
        zoom_label.setFont(QFont("Arial", 9, QFont.Bold))
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setFixedSize(self.ZOOM_BUTTON_SIZE, 22)
        zoom_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 160); border-radius: 4px;")
        panel_layout.addWidget(zoom_in)
        panel_layout.addWidget(zoom_label)
        panel_layout.addWidget(zoom_out)
        # Fixed size keeps the placement maths correct before the panel is first shown.
        panel.setFixedSize(self.ZOOM_BUTTON_SIZE + 8, self.ZOOM_BUTTON_SIZE * 2 + 22 + 16)
        parent_label.zoom_panel = panel
        parent_label.position_zoom_panel()
        panel.show()
        return panel, zoom_in, zoom_out, zoom_label

    def create_feed_tab(self, side: str) -> QWidget:
        """Build one camera feed tab. The tab only owns the camera view; the
        Scan Controls, LED Controls, Current Detection and Detection Log panels
        are the shared widgets attached by attach_shared_panels()."""
        tab = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        tab.setLayout(layout)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)
        # Feed column takes the dominant share of the width; the info column is a
        # compact fixed-width strip on the right so the camera feed stays large.
        layout.addLayout(left_panel, stretch=4)

        # A single feed per tab. It expands to fill all the space the info column
        # does not claim, so the live picture dominates the 7-inch screen.
        camera_label = CameraLabel()
        camera_label.setMinimumSize(320, 240)
        camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Light text: the default black is invisible against the black feed panel.
        camera_label.setStyleSheet(
            "border: 2px solid #333; background-color: #000; border-radius: 5px; "
            "color: #999; font-size: 14px; font-weight: bold;")
        camera_label.setAlignment(Qt.AlignCenter)
        # No placeholder text: the live preview fills the panel as soon as the
        # cameras open. A camera that is not plugged in simply shows black.
        camera_label.setText("")
        left_panel.addWidget(camera_label, stretch=1)

        camera_title = QLabel(f"{side.capitalize()} Camera")
        camera_title.setFont(QFont("Arial", 10, QFont.Bold))
        camera_title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 3px; border-radius: 3px;")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setParent(camera_label)
        camera_title.move(8, 8)
        camera_title.show()

        panel, zoom_in, zoom_out, zoom_label = self._build_zoom_overlay(camera_label, side)
        camera_label.zoom_panel = panel
        # Connect drag-to-pan so the user can navigate the zoomed feed without
        # the layout shifting or the zoom overlay jumping.
        camera_label.pan_requested.connect(lambda dx, dy, s=side: self.pan_feed(s, dx, dy))
        camera_label.setCursor(Qt.OpenHandCursor)
        setattr(self, f"{side}_camera_label", camera_label)
        setattr(self, f"{side}_camera_title", camera_title)
        setattr(self, f"{side}_zoom_panel", panel)
        setattr(self, f"{side}_zoom_in_button", zoom_in)
        setattr(self, f"{side}_zoom_out_button", zoom_out)
        setattr(self, f"{side}_zoom_label", zoom_label)

        # Hosts for the shared panels: controls under the feed, info on the right
        controls_host = QVBoxLayout()
        controls_host.setSpacing(4)
        left_panel.addLayout(controls_host, stretch=0)

        info_host = QVBoxLayout()
        info_host.setSpacing(4)
        layout.addLayout(info_host, stretch=0)

        self.controls_hosts[side] = controls_host
        self.info_hosts[side] = info_host
        return tab

    def attach_shared_panels(self, side: str):
        """Move the single shared controls/info panels into the given feed tab so
        both tabs always show identical controls and detection state."""
        previous = getattr(self, '_attached_side', None)
        if previous == side:
            return
        if previous is not None:
            self.controls_hosts[previous].removeWidget(self.shared_controls_widget)
            self.info_hosts[previous].removeWidget(self.shared_info_widget)
        self.controls_hosts[side].addWidget(self.shared_controls_widget)
        self.info_hosts[side].addWidget(self.shared_info_widget)
        self.shared_controls_widget.show()
        self.shared_info_widget.show()
        self._attached_side = side

    def on_tab_changed(self, index: int):
        widget = self.tab_widget.widget(index)
        if widget is getattr(self, 'left_feed_tab', None):
            self.attach_shared_panels('left')
            # Reposition the zoom overlay on the now-visible tab; the hidden tab
            # never gets a resize event so the overlay would sit in the corner.
            self.left_camera_label.position_zoom_panel()
        elif widget is getattr(self, 'right_feed_tab', None):
            self.attach_shared_panels('right')
            self.right_camera_label.position_zoom_panel()

    def build_shared_panels(self):
        self.controls_hosts = {}
        self.info_hosts = {}
        self._attached_side = None

        # Scan and LED controls sit side by side in a single strip so the strip
        # stays one row tall and the camera feed keeps the rest of the tab.
        self.shared_controls_widget = QWidget()
        # Fixed height so the controls strip never expands vertically and eats
        # camera feed space. Tall enough for a 52px touch button plus margins.
        self.shared_controls_widget.setFixedHeight(78)
        self.shared_controls_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self.shared_controls_widget.setLayout(controls_layout)

        self.shared_info_widget = QWidget()
        # Compact fixed-width info column so the camera feed keeps the dominant
        # share of the tab. Wide enough for the Current Detection / System Log
        # text to be readable on the 7-inch panel without overpowering the feed.
        self.shared_info_widget.setFixedWidth(self.INFO_PANEL_WIDTH)
        self.shared_info_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        self.shared_info_widget.setLayout(info_layout)

        # Scan controls below the camera feed
        scan_group = QGroupBox("Scan Controls")
        scan_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        # Side by side so the controls strip stays one row tall and the feed keeps
        # the rest of the tab.
        scan_layout = QHBoxLayout()
        scan_layout.setContentsMargins(6, 4, 6, 4)
        scan_layout.setSpacing(4)
        
        self.start_button = self._make_touch_button(
            "Start Scan", "#4CAF50", "#45a049", self.toggle_scan, pressed="#3d8b40")
        scan_layout.addWidget(self.start_button)

        scan_group.setLayout(scan_layout)
        controls_layout.addWidget(scan_group, stretch=2)
        
        # LED controls beside the scan controls
        led_group = QGroupBox("LED Controls")
        led_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        led_layout = QHBoxLayout()
        led_layout.setContentsMargins(6, 4, 6, 4)
        led_layout.setSpacing(4)
        
        self.red_light_button = self._make_touch_button(
            "Red Light", "#f44336", "#d32f2f", self.set_red_light, pressed="#b71c1c")
        led_layout.addWidget(self.red_light_button)
        
        self.white_light_button = self._make_touch_button(
            "White Light", "#ffffff", "#e0e0e0", self.set_white_light,
            pressed="#cccccc", color="black", border="2px solid #333")
        led_layout.addWidget(self.white_light_button)
        
        self.led_off_button = self._make_touch_button(
            "LEDs Off", "#333333", "#555555", self.set_leds_off, pressed="#111111")
        led_layout.addWidget(self.led_off_button)
        
        led_group.setLayout(led_layout)
        controls_layout.addWidget(led_group, stretch=5)
        
        # Current detection info
        current_info_group = QGroupBox("Current Detection")
        current_info_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        current_info_layout = QVBoxLayout()
        current_info_layout.setSpacing(4)
        current_info_layout.setContentsMargins(6, 4, 6, 6)

        self.count_label = QLabel("Weevil Count: --")
        self.count_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("padding: 6px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.count_label)

        # Confidence and temperature stacked vertically so the labels are not
        # jammed side by side in the narrow info column.
        self.confidence_label = QLabel("Avg Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 10))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("padding: 5px; background-color: #e8f5e9; border-radius: 5px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.confidence_label)

        self.temp_label = QLabel("Temperature: --°C")
        self.temp_label.setFont(QFont("Arial", 10))
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setStyleSheet("padding: 5px; background-color: #e3f2fd; border-radius: 5px; border: 1px solid #bbdefb;")
        current_info_layout.addWidget(self.temp_label)
        
        self.recommendation_label = QLabel("Recommendation: --")
        self.recommendation_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setMinimumWidth(0)
        self.recommendation_label.setStyleSheet("color: #0066cc; padding: 5px; background-color: #fff3e0; border-radius: 5px; border: 1px solid #ffe0b2;")
        current_info_layout.addWidget(self.recommendation_label)

        current_info_group.setLayout(current_info_layout)
        info_layout.addWidget(current_info_group)
        
        # Detection model runtime performance - inference time and detection rate only.
        # The static model/backend details are logged at startup instead of taking up
        # room in the narrow side panel.
        model_group = QGroupBox("Detection Model")
        model_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        # Stacked vertically so the two readouts are not jammed side by side.
        model_layout = QVBoxLayout()
        model_layout.setSpacing(4)
        model_layout.setContentsMargins(6, 4, 6, 6)

        self.inference_label = QLabel("Inference: --")
        self.inference_label.setFont(QFont("Consolas", 10, QFont.Bold))
        self.inference_label.setAlignment(Qt.AlignCenter)
        self.inference_label.setStyleSheet("padding: 5px; background-color: #fff8e1; border-radius: 5px; border: 1px solid #ffecb3; color: #5d4037;")
        model_layout.addWidget(self.inference_label)

        self.rate_label = QLabel("Rate: --")
        self.rate_label.setFont(QFont("Consolas", 10, QFont.Bold))
        self.rate_label.setAlignment(Qt.AlignCenter)
        self.rate_label.setStyleSheet("padding: 5px; background-color: #fff8e1; border-radius: 5px; border: 1px solid #ffecb3; color: #5d4037;")
        model_layout.addWidget(self.rate_label)
        
        model_group.setLayout(model_layout)
        info_layout.addWidget(model_group)
        
        # System log display - compact
        log_group = QGroupBox("System Log")
        log_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(6, 4, 6, 6)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(60)
        self.log_text.setMaximumHeight(140)
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;")
        self._make_touch_scrollable(self.log_text)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        info_layout.addWidget(log_group, stretch=1)

    def setup_scan_history_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.scan_history_tab.setLayout(layout)

        # Touch-friendly styling: tall rows, generous padding and a clearly
        # highlighted selection so a finger tap is easy to aim and easy to confirm.
        table_style = f"""
            QTableWidget {{
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: white;
                gridline-color: #eee;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 10px 8px;
                border-bottom: 1px solid #eee;
            }}
            QTableWidget::item:selected {{
                background-color: #1976D2;
                color: white;
            }}
            QHeaderView::section {{
                background-color: #f5f5f5;
                padding: 10px 6px;
                border: 1px solid #ddd;
                font-weight: bold;
                font-size: 12px;
                color: #333;
            }}
            QScrollBar:vertical {{
                background: #f0f0f0;
                width: {self.TOUCH_SCROLLBAR_SIZE}px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #9e9e9e;
                border-radius: {self.TOUCH_SCROLLBAR_SIZE // 2}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:pressed {{ background: #616161; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar:horizontal {{
                background: #f0f0f0;
                height: {self.TOUCH_SCROLLBAR_SIZE}px;
                margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: #9e9e9e;
                border-radius: {self.TOUCH_SCROLLBAR_SIZE // 2}px;
                min-width: 40px;
            }}
            QScrollBar::handle:horizontal:pressed {{ background: #616161; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """

        splitter = QSplitter(Qt.Horizontal)
        # A thin desktop splitter handle is impossible to grab with a fingertip.
        splitter.setHandleWidth(12)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #cfd8dc; border-radius: 4px; } "
            "QSplitter::handle:pressed { background-color: #90a4ae; }")
        layout.addWidget(splitter, stretch=1)
        
        # Left: stored scans
        scans_group = QGroupBox("Stored Scans")
        scans_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        scans_layout = QVBoxLayout()
        scans_layout.setContentsMargins(4, 4, 4, 4)
        
        self.scan_table = QTableWidget()
        self.scan_table.setColumnCount(7)
        self.scan_table.setHorizontalHeaderLabels(
            ["Scan ID", "Start Time", "End Time", "Scan Time", "Max Count", "Avg Temp", "Images"])
        self.scan_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.scan_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.scan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.scan_table.setSelectionMode(QTableWidget.SingleSelection)
        self.scan_table.setStyleSheet(table_style)
        self._make_touch_scrollable(self.scan_table)
        # Tall rows and no row numbers: the vertical header is dead space on a
        # touchscreen and the scan id already identifies the row.
        self.scan_table.verticalHeader().setDefaultSectionSize(self.TOUCH_ROW_HEIGHT)
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.itemSelectionChanged.connect(self.on_scan_selected)
        scans_layout.addWidget(self.scan_table)
        
        scans_group.setLayout(scans_layout)
        splitter.addWidget(scans_group)
        
        # Right: images and detection log stored for the selected scan
        detail_group = QGroupBox("Stored Detection Log and Images")
        detail_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(5)
        
        self.image_preview_label = QLabel("Select a scan to preview its stored images")
        self.image_preview_label.setMinimumHeight(140)
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setStyleSheet("border: 2px solid #333; background-color: #000; color: #999; border-radius: 5px; font-size: 12px;")
        detail_layout.addWidget(self.image_preview_label, stretch=2)
        
        self.image_list = QListWidget()
        # Two finger-sized rows are visible at a time; the rest is scrolled.
        self.image_list.setFixedHeight(self.TOUCH_ROW_HEIGHT * 2 + 8)
        self.image_list.setStyleSheet(f"""
            QListWidget {{
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }}
            QListWidget::item {{
                min-height: {self.TOUCH_ROW_HEIGHT}px;
                padding: 4px 8px;
                border-bottom: 1px solid #eee;
            }}
            QListWidget::item:selected {{ background-color: #1976D2; color: white; }}
            QScrollBar:vertical {{
                background: #f0f0f0;
                width: {self.TOUCH_SCROLLBAR_SIZE}px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #9e9e9e;
                border-radius: {self.TOUCH_SCROLLBAR_SIZE // 2}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:pressed {{ background: #616161; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        self._make_touch_scrollable(self.image_list)
        self.image_list.currentRowChanged.connect(self.on_history_image_selected)
        detail_layout.addWidget(self.image_list, stretch=0)
        
        self.history_log_table = QTableWidget()
        self.history_log_table.setColumnCount(4)
        self.history_log_table.setHorizontalHeaderLabels(["Timestamp", "Count", "Temp", "Recommendation"])
        self.history_log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_log_table.setStyleSheet(table_style)
        self._make_touch_scrollable(self.history_log_table)
        self.history_log_table.verticalHeader().setDefaultSectionSize(self.TOUCH_ROW_HEIGHT)
        self.history_log_table.verticalHeader().setVisible(False)
        detail_layout.addWidget(self.history_log_table, stretch=2)
        
        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)
        splitter.setSizes([520, 420])
        # Neither pane may be collapsed to zero by a stray drag, which on a
        # touchscreen would be very easy to do and hard to undo.
        splitter.setChildrenCollapsible(False)
        
        # Actions - the merged tab keeps Refresh, Email Selected Scan Report and
        # Open Scans Folder. The manual "Export Zip from Database" button was removed
        # because the email already delivers the same archive automatically.
        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        
        def make_button(text, color, hover, handler):
            button = QPushButton(text)
            button.setMinimumHeight(self.TOUCH_BUTTON_SIZE)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-size: 13px;
                    font-weight: bold;
                    border-radius: 6px;
                    padding: 6px;
                }}
                QPushButton:hover {{ background-color: {hover}; }}
                QPushButton:pressed {{ background-color: {hover}; }}
                QPushButton:disabled {{ background-color: #bdbdbd; }}
            """)
            button.clicked.connect(handler)
            button_row.addWidget(button)
            return button
        
        self.refresh_history_button = make_button("Refresh", "#2196F3", "#1976D2", self.refresh_scan_history)
        self.delete_scan_button = make_button("Delete Selected Scan", "#E53935", "#C62828",
                                              self.delete_selected_scan)
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
        
        # Initialize temperature sensor (DS18B20 over 1-Wire, polled in its own thread)
        if self.temp_sensor.initialize():
            status = self.temp_sensor.get_status()
            self.log_message(f"DS18B20 temperature sensor ready: {status['device_id']} "
                             f"(polling every {status['poll_interval']:g}s)")
        else:
            self.log_message(f"DS18B20 not available - {self.temp_sensor.last_error}")
        
        # Initialize LED controller (WS2813; SPI on Pi 5, rpi_ws281x on Pi 4 and older)
        self.led_controller.initialize()
        led_status = self.led_controller.get_status()
        if led_status['backend'] == 'simulation':
            self.log_message(f"WS2813 LEDs in simulation mode - {led_status['error']}")
        else:
            self.log_message(f"WS2813 LEDs ready: {led_status['led_count']} pixels via "
                             f"{led_status['backend']}, brightness {led_status['brightness']}")
        
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

        # Start live camera preview immediately so both feeds are visible before
        # the user clicks Start Scan. Cameras that are not plugged in simply stay
        # black - no "No Signal" placeholder is shown.
        self.start_preview()

    def start_preview(self):
        """Open the cameras and begin a lightweight preview loop (no detection).

        Called at app launch and again after every scan stops so the feeds stay
        live. The cameras are opened once and left open; the detection thread
        reuses the same camera_manager when a scan starts.
        """
        if self.preview_thread is not None:
            return
        if not self.camera_manager.is_running():
            if not self.camera_manager.start():
                self.log_message("No camera device available for live preview")
                return
        self.preview_thread = PreviewThread(self.camera_manager)
        self.preview_thread.frame_ready_left.connect(self.update_left_preview)
        self.preview_thread.frame_ready_right.connect(self.update_right_preview)
        self.preview_thread.camera_status.connect(self.update_camera_status)
        self.preview_thread.start()
        self.log_message("Live camera preview started")

    def stop_preview(self):
        """Stop the preview loop. The cameras themselves stay open so the
        detection thread can take over without re-opening the devices."""
        if self.preview_thread is not None:
            self.preview_thread.stop()
            self.preview_thread = None

    def update_left_preview(self, frame: np.ndarray):
        """Render a raw camera frame (no detection annotations) for live preview."""
        self.latest_annotated_left = frame
        self._render_frame_to_label(frame, self.left_camera_label, self.left_zoom,
                                    self.left_pan_x, self.left_pan_y)

    def update_right_preview(self, frame: np.ndarray):
        """Render a raw camera frame (no detection annotations) for live preview."""
        self.latest_annotated_right = frame
        self._render_frame_to_label(frame, self.right_camera_label, self.right_zoom,
                                    self.right_pan_x, self.right_pan_y)

    def update_model_display(self):
        """Report which YOLOv11n weights and backend are loaded to the Detection Log.

        The Detection Model panel itself only shows the live Inference and Rate
        figures, so the static details go to the log instead of the side panel.
        """
        info = self.detector.get_model_info()
        status = "loaded" if info['initialized'] else "NOT LOADED"
        classes = ', '.join(str(c) for c in info['classes']) or 'none'
        metrics = info['metrics']
        self.log_message(
            f"Model: {info['architecture']} - {info['model_name']} ({status})   |   "
            f"Backend: {info['backend']}   |   Device: {info['device']}   |   "
            f"Input: {info['imgsz']}px (trained {info['trained_imgsz']}px)   |   "
            f"Classes: {classes}   |   Confidence >= {info['confidence_threshold']}   |   "
            f"IoU {info['iou_threshold']}   |   Cycle every {self.detection_interval_ms} ms   |   "
            f"Training ({metrics['run']}, {metrics['epochs']} epochs): "
            f"mAP50 {metrics['mAP50']:.3f}, mAP50-95 {metrics['mAP50-95']:.3f}, "
            f"P {metrics['precision']:.3f}, R {metrics['recall']:.3f}")
        
        warning = info['warning']
        if not info['initialized']:
            warning = warning or "Model failed to load - no detections will be recorded."
        if warning:
            self.log_message(f"MODEL WARNING: {warning}")

    def update_performance_stats(self, rate_per_second: float, avg_inference_ms: float):
        self.inference_label.setText(f"Inference: {avg_inference_ms:.0f}ms")
        self.rate_label.setText(f"Rate: {rate_per_second:.1f}/s")

    def toggle_scan(self):
        if self.is_scanning:
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self):
        # Stop the lightweight preview loop so the detection thread can take
        # over the same camera feeds without contention.
        self.stop_preview()
        if not self.camera_manager.start():
            self.log_message("Failed to start cameras - no capture device could be opened")
            return

        # TEMPORARY TEST MODE: skip live-signal validation so the scan starts
        # with whatever camera opened, even if frames are blank/frozen.
        left_ok = self.camera_manager.left_camera.is_running()
        right_ok = self.camera_manager.right_camera.is_running()

        if not left_ok and not right_ok:
            self.log_message("Scan cancelled - no camera device could be opened")
            self.update_camera_status(False, False)
            QMessageBox.warning(
                self, "No Camera",
                "No camera device could be opened, so the scan was not started.")
            # Resume live preview if any camera is still available
            self.start_preview()
            return

        if left_ok and right_ok:
            self.log_message("Both cameras opened (signal validation bypassed for testing)")
        else:
            self.log_message(f"Scanning with the "
                             f"{'left' if left_ok else 'right'} camera only "
                             f"(signal validation bypassed for testing)")
        self.update_camera_status(left_ok, right_ok)
        
        # Create scan folder with timestamp
        # Scan ID format: scan_YYYY-MM-DD_HH-MM-SS (readable, sortable, Year-Month-Date order)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        self.detection_thread.camera_status.connect(self.update_camera_status)
        self.detection_thread.start()
        
        # Reset image/log tracking so the previous scan's timings do not leak in
        self.last_image_capture_time = None
        self.last_log_time = None
        self.last_detection_log_time = None
        self._last_temp_sample_id = None
        
        self.is_scanning = True
        self.start_button.setText("Stop Scan")
        self.start_button.setStyleSheet(
            self._touch_button_style("#f44336", "#d32f2f", pressed="#b71c1c"))
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
        self.start_button.setStyleSheet(
            self._touch_button_style("#4CAF50", "#45a049", pressed="#3d8b40"))
        # Reset the per-frame detection display, but show the final collective
        # recommendation based on the scan's max weevil count. "Activate Mix" /
        # "Unload Rice" are post-scan decisions, not per-frame readings.
        self.count_label.setText(f"Weevil Count: {self.scan_max_count}")
        self.confidence_label.setText("Avg Confidence: --")
        final_rec = self.logger.generate_recommendation(
            self.scan_max_count, self.is_after_mixing, is_final=True)
        self.recommendation_label.setText(f"Recommendation: {final_rec}")
        self.status_label.setText("Ready")
        self.latest_annotated_left = None
        self.latest_annotated_right = None
        self.log_message("Scan stopped and saved")

        # Keep the cameras open and restart the preview loop so the feeds stay
        # live after the scan ends. The cameras are only stopped on app exit.
        self.start_preview()

        if scan_id:
            self.email_scan_report(scan_id)
            self.refresh_scan_history()

    def email_scan_report(self, scan_id: str) -> Optional[str]:
        """Build the report archive from the database and email it automatically.

        Called when a scan stops (either the user clicked Stop Scan or the 3-minute
        protocol timer elapsed). Everything in the archive - the detection log CSV
        and the captured rice weevil images - is read back out of SQLite.
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
        """Reload the Detection Logs tab from the database."""
        try:
            scans = self.db.get_scan_overview()
        except Exception as e:
            self.log_message(f"Error reading scan history: {e}")
            return

        self.scan_table.setRowCount(0)
        for scan in scans:
            row = self.scan_table.rowCount()
            self.scan_table.insertRow(row)
            avg_temp = scan.get('avg_temperature_celsius')
            # Compute scan duration from start_time and end_time stored in the DB.
            start_str = scan.get('start_time', '')
            end_str = scan.get('end_time', '')
            scan_time = "--"
            try:
                if start_str and end_str:
                    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                    delta = end_dt - start_dt
                    total_secs = int(delta.total_seconds())
                    if total_secs >= 0:
                        mins, secs = divmod(total_secs, 60)
                        scan_time = f"{mins}m {secs}s"
            except (ValueError, TypeError):
                pass
            values = [
                scan.get('scan_id', ''),
                start_str,
                end_str,
                scan_time,
                str(scan.get('max_weevil_count', 0)),
                f"{avg_temp:.1f}°C" if isinstance(avg_temp, (int, float)) else "N/A",
                f"{scan.get('image_count', 0)} ({(scan.get('image_total_bytes') or 0) / 1024:.0f} KB)",
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

    def delete_selected_scan(self):
        """Delete the selected stored scan from the database and disk.

        Removing the scan row also removes its detections, stored images and report
        records, so the Stored Detection Log and Images pane is cleared and reloaded
        to stay consistent with what is actually stored.
        """
        scan_id = self.selected_scan_id()
        if not scan_id:
            QMessageBox.information(self, "Delete Scan", "Select a stored scan first.")
            return
        if self.is_scanning and scan_id == self.current_scan_id:
            QMessageBox.warning(self, "Delete Scan",
                                "Stop the running scan before deleting it.")
            return
        
        confirm = QMessageBox.question(
            self, "Delete Scan",
            f"Delete {scan_id}?\n\nIts detection log, stored images, recordings and "
            f"report archive are removed permanently.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        
        # Capture what is about to be lost so the notification email can describe it
        try:
            scan = self.db.get_scan_by_id(scan_id) or {}
            detection_count = len(self.db.get_detections_by_scan(scan_id))
            image_count = self.db.get_scan_image_count(scan_id)
        except Exception as e:
            self.log_message(f"Could not read {scan_id} before deleting: {e}")
            scan, detection_count, image_count = {}, 0, 0
        
        try:
            deleted = self.db.delete_scan(scan_id)
        except Exception as e:
            self.log_message(f"Error deleting {scan_id} from the database: {e}")
            QMessageBox.critical(self, "Delete Scan", f"Could not delete {scan_id}:\n{e}")
            return
        
        # Keep the file system in step with the database
        for path in (os.path.join(self.previous_scans_dir, scan_id),
                     os.path.join(self.reports_dir, f"{scan_id}_report.zip")):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except OSError as e:
                self.log_message(f"Could not remove {path}: {e}")
        
        self.log_message(f"Deleted scan {scan_id}" if deleted
                         else f"Scan {scan_id} was not found in the database")
        
        if deleted:
            avg_temp = scan.get('avg_temperature_celsius')
            temp_str = f"{avg_temp:.1f} C" if isinstance(avg_temp, (int, float)) else "N/A"
            subject = f"Anilag Scan Deleted - {scan_id}"
            body = (
                f"Anilag Rice Weevil Detection System\n"
                f"====================================\n\n"
                f"A stored scan was deleted by the user from the Detection Logs tab.\n\n"
                f"Scan ID: {scan_id}\n"
                f"Deletion timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Scan start time: {scan.get('start_time', 'unknown')}\n"
                f"Scan end time: {scan.get('end_time', 'unknown')}\n"
                f"Max weevil count: {scan.get('max_weevil_count', 'unknown')}\n"
                f"Average temperature: {temp_str}\n"
                f"Detection log entries removed: {detection_count}\n"
                f"Stored images removed: {image_count}\n\n"
                f"The scan recordings and report archive were removed from storage as well. "
                f"This deletion is permanent.\n\n"
                f"This is an automated notification from the Anilag detection system.")
            # Send directly via send_email so the notification is not gated by the
            # EMAIL_ACTIVITY_ALERTS opt-in (which defaults to false).  Scan deletion is
            # an important data-loss event that the admin should always be told about.
            self.send_email_async(f"Scan Deleted {scan_id}",
                                  self.email_notifier.send_email, subject, body)
        
        # Clear the detail pane; refresh_scan_history re-selects a remaining scan
        # (or leaves the pane empty when none are left).
        self.image_list.clear()
        self.history_log_table.setRowCount(0)
        self.image_preview_label.setPixmap(QPixmap())
        self.image_preview_label.setText("Select a scan to preview its stored images")
        self.refresh_scan_history()

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
        
        if not self.current_scan_id:
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
            "recommendation": self.logger.generate_recommendation(self.scan_max_count, self.is_after_mixing, is_final=True),
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
            "temperature_sensor": self.temp_sensor.get_status(),
            "led_controller": self.led_controller.get_status(),
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
        
        # Also drop a copy next to the recordings for offline inspection. This is
        # best-effort: losing the scan folder (e.g. the external SSD unmounting) must
        # not cost us the database record written above.
        if self.current_scan_folder:
            try:
                metadata_path = os.path.join(self.current_scan_folder, "scan_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            except OSError as e:
                self.log_message(f"Could not write scan_metadata.json ({e}); "
                                 "the database record was saved regardless")
        
        self.log_message(f"Scan metadata saved to database ({self.scan_detection_count} log entries, "
                         f"{image_count} images stored)")
        return metadata

    def set_red_light(self):
        applied = self.led_controller.set_red()
        suffix = "" if applied else " (simulated - no LED hardware)"
        self.log_message(f"Red light activated{suffix}")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "Red light activated to lure rice weevils")

    def set_white_light(self):
        applied = self.led_controller.set_white()
        suffix = "" if applied else " (simulated - no LED hardware)"
        self.log_message(f"White light activated{suffix}")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "White light activated for detection")

    def set_leds_off(self):
        applied = self.led_controller.off()
        suffix = "" if applied else " (simulated - no LED hardware)"
        self.log_message(f"LEDs turned off{suffix}")
        self.send_email_async("LED Control", self.email_notifier.send_activity_log,
                              "LED Control", "LEDs turned off")

    def adjust_zoom(self, side: str, direction: int):
        """Adjust the digital zoom factor for the left or right camera feed.

        direction = +1 zooms in, -1 zooms out. The zoom is display-only: the
        recorded video and stored images always keep the full frame.
        """
        attr = f"{side}_zoom"
        current = getattr(self, attr)
        new = round(current + direction * self.zoom_step, 2)
        new = max(self.zoom_min, min(self.zoom_max, new))
        setattr(self, attr, new)
        # Reset pan to centre whenever the zoom level changes so the crop stays
        # within bounds.
        setattr(self, f"{side}_pan_x", 0.0)
        setattr(self, f"{side}_pan_y", 0.0)
        label = getattr(self, f"{side}_zoom_label")
        label.setText(f"{new:.1f}x")
        # Re-render the last frame at the new zoom level if we have one
        latest = getattr(self, f"latest_annotated_{'left' if side == 'left' else 'right'}")
        if latest is not None:
            target_label = self.left_camera_label if side == 'left' else self.right_camera_label
            self._render_frame_to_label(latest, target_label, new,
                                        getattr(self, f"{side}_pan_x"),
                                        getattr(self, f"{side}_pan_y"))

    def pan_feed(self, side: str, delta_screen_x: float, delta_screen_y: float):
        """Pan the zoomed feed in response to a drag on the camera label.

        Screen-pixel deltas are converted to frame-pixel offsets using the
        current zoom factor, then clamped so the crop window never leaves the
        frame. The feed is re-rendered immediately so dragging feels smooth.
        """
        zoom = getattr(self, f"{side}_zoom")
        if zoom <= 1.0:
            return  # No panning when not zoomed in
        # Convert screen drag to frame-pixel drag (zoomed view means 1 screen
        # pixel = 1/zoom frame pixels).
        frame_dx = delta_screen_x / zoom
        frame_dy = delta_screen_y / zoom
        new_x = getattr(self, f"{side}_pan_x") - frame_dx
        new_y = getattr(self, f"{side}_pan_y") - frame_dy
        # Clamp so the crop window stays inside the frame. The crop is centred
        # at (w/2 + pan_x, h/2 + pan_y) with size (w/zoom, h/zoom), so the pan
        # range is +/- (w/2 - crop_w/2).
        latest = getattr(self, f"latest_annotated_{'left' if side == 'left' else 'right'}")
        if latest is None:
            return
        h, w = latest.shape[:2]
        crop_w = w / zoom
        crop_h = h / zoom
        max_x = (w - crop_w) / 2
        max_y = (h - crop_h) / 2
        new_x = max(-max_x, min(max_x, new_x))
        new_y = max(-max_y, min(max_y, new_y))
        setattr(self, f"{side}_pan_x", new_x)
        setattr(self, f"{side}_pan_y", new_y)
        target_label = self.left_camera_label if side == 'left' else self.right_camera_label
        self._render_frame_to_label(latest, target_label, zoom, new_x, new_y)

    def _render_frame_to_label(self, frame: np.ndarray, label, zoom: float,
                               pan_x: float = 0.0, pan_y: float = 0.0):
        """Render a BGR frame to a QLabel, applying the given digital zoom and pan.

        Zoom is a center crop: at 2.0x, the central half of the frame (width and
        height) is scaled up to fill the label. pan_x/pan_y (in frame pixels)
        shift the crop centre so the user can drag to navigate the zoomed view.
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if zoom > 1.0:
            h, w = rgb_image.shape[:2]
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            # Centre the crop, then apply pan offset (clamped to frame bounds)
            cx = w / 2 + pan_x
            cy = h / 2 + pan_y
            x0 = int(max(0, min(w - crop_w, cx - crop_w / 2)))
            y0 = int(max(0, min(h - crop_h, cy - crop_h / 2)))
            rgb_image = rgb_image[y0:y0 + crop_h, x0:x0 + crop_w]
            # Slicing produces a non-contiguous view; QImage needs a contiguous buffer
            rgb_image = np.ascontiguousarray(rgb_image)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        # Fill the whole panel: scale up until both dimensions are covered, then
        # trim the centre to the panel size. This maximizes the camera input -
        # no black letterbox bars - regardless of the panel's aspect ratio.
        target = label.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled_pixmap = pixmap.scaled(target, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        if scaled_pixmap.width() > target.width() or scaled_pixmap.height() > target.height():
            x0 = (scaled_pixmap.width() - target.width()) // 2
            y0 = (scaled_pixmap.height() - target.height()) // 2
            scaled_pixmap = scaled_pixmap.copy(x0, y0, target.width(), target.height())
        label.setPixmap(scaled_pixmap)

    def update_left_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        self.latest_annotated_left = annotated_frame
        
        # Write to video file if recording (always full frame, not zoomed)
        if self.video_writer_left and self.video_writer_left.isOpened():
            self.video_writer_left.write(annotated_frame)

        # Display with current zoom level (display-only; recording keeps full frame)
        self._render_frame_to_label(annotated_frame, self.left_camera_label, self.left_zoom,
                                    self.left_pan_x, self.left_pan_y)

    def update_right_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        self.latest_annotated_right = annotated_frame

        # Write to video file if recording (always full frame, not zoomed)
        if self.video_writer_right and self.video_writer_right.isOpened():
            self.video_writer_right.write(annotated_frame)

        # Display with current zoom level (display-only; recording keeps full frame)
        self._render_frame_to_label(annotated_frame, self.right_camera_label, self.right_zoom,
                                    self.right_pan_x, self.right_pan_y)

    def update_camera_status(self, left_live: bool, right_live: bool):
        """Clear a feed that has no live signal so no stale picture (and no stale
        bounding boxes) is left on screen for a camera that is not delivering.
        No placeholder text is shown - the panel simply stays black until the
        camera delivers frames again."""
        for live, label, side in ((left_live, self.left_camera_label, 'left'),
                                  (right_live, self.right_camera_label, 'right')):
            if live:
                continue
            label.setPixmap(QPixmap())
            label.setText("")
            setattr(self, f"latest_annotated_{side}", None)

    def update_detection(self, count: int, confidence: float, activity: str):
        # Only show count/confidence during an active scan with cameras running.
        # Before the scan starts or after it stops, the labels stay at "--" so the
        # UI does not display a phantom count from a non-existent camera feed.
        if self.is_scanning:
            self.count_label.setText(f"Weevil Count: {count}")
            self.confidence_label.setText(
                f"Avg Confidence: {confidence * 100:.1f}%" if confidence else "Avg Confidence: --")
        
        # Get temperature (cached; the DS18B20 is polled in its own thread)
        sample_id, temperature = self.temp_sensor.read_sample()
        
        # Track scan metadata
        if self.is_scanning:
            if count > self.scan_max_count:
                self.scan_max_count = count
            # Detection cycles run far faster than the sensor converts, so only count
            # each physical conversion once - otherwise the scan average is weighted by
            # how long a reading happened to sit in the cache.
            if temperature is not None and sample_id != self._last_temp_sample_id:
                self.scan_temp_readings.append(temperature)
                self._last_temp_sample_id = sample_id
        
        current_time = datetime.now()
        # "Significant" means a noticeably high reading, not merely a non-zero one.
        is_significant = count >= self.high_weevil_threshold
        
        # Capture images for any non-zero count, rate-limited by the cooldown.  Previously
        # this only fired for high counts (>= HIGH_WEEVIL_THRESHOLD), so a scan that saw
        # 1-4 weevils stored zero images and the zip archive had an empty detected_images/
        # folder.  High-count frames are still prioritised by the log marker below.
        should_capture = (self.is_scanning and count > 0 and
                         (self.last_image_capture_time is None or
                          (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_cooldown))
        
        # Log every reading that contains weevils, not just high counts - otherwise a
        # scan that only ever sees 1-4 weevils stores nothing but the sparse baseline
        # rows and the stored detection log looks empty. Rate-limited so a sustained
        # detection does not flood the table. The baseline row still guarantees quiet
        # scans record that the equipment was running and found nothing.
        should_log_detection = count > 0 and (
            self.last_detection_log_time is None or
            (current_time - self.last_detection_log_time).total_seconds() >= self.detection_log_interval_seconds)
        should_log_baseline = (self.last_log_time is None or
                               current_time - self.last_log_time >= timedelta(seconds=self.log_interval_seconds))
        should_log = should_log_detection or should_log_baseline
        
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
            marker = "HIGH COUNT " if is_significant else ""
            self.log_message(f"{marker}{log_entry.timestamp} - Count: {count}, Temp: {temp_str}, "
                             f"Rec: {log_entry.recommendation}")
            self.last_log_time = current_time
            if should_log_detection:
                self.last_detection_log_time = current_time
    
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
        prefix = "high_weevil" if count >= self.high_weevil_threshold else "weevil"
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
            if count >= self.high_weevil_threshold:
                self.log_message(f"High weevil count ({count} >= {self.high_weevil_threshold}) - "
                                 f"stored {len(saved)} image(s) in database")
            else:
                self.log_message(f"Weevil count {count} - "
                                 f"stored {len(saved)} image(s) in database")
        return saved

    def update_temperature(self):
        """Refresh the temperature readout. Non-blocking: the DS18B20 is polled in
        its own thread, so this only reads the cached value."""
        temperature = self.temp_sensor.read_temperature()
        status = self.temp_sensor.get_status()
        
        if temperature is not None:
            self.temp_label.setText(f"Temperature: {temperature:.1f}°C")
            self.temp_label.setStyleSheet(
                "padding: 8px; background-color: #e3f2fd; border-radius: 5px; border: 1px solid #bbdefb;")
        elif not status['available']:
            self.temp_label.setText("Temperature: sensor not detected")
            self.temp_label.setStyleSheet(
                "padding: 8px; background-color: #ffebee; border-radius: 5px; border: 1px solid #ef9a9a; color: #b71c1c;")
        else:
            self.temp_label.setText("Temperature: no recent reading")
            self.temp_label.setStyleSheet(
                "padding: 8px; background-color: #fff8e1; border-radius: 5px; border: 1px solid #ffecb3; color: #8d6e00;")
        
        # Surface a sensor fault once rather than on every tick
        error = status.get('last_error')
        if error and error != getattr(self, '_last_temp_error', None):
            self.log_message(f"Temperature sensor: {error}")
        self._last_temp_error = error
    
    def update_clock(self):
        now = datetime.now()
        # Format: June 25, 2026 9:31:45 PM
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        # The header clock stays visible on every tab, so a separate date/time
        # panel on Live Feed would only duplicate it.
        self.header_clock_label.setText(f"{date_str}   {time_str}")

    def log_message(self, message: str):
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_console_line(self, line: str):
        """Slot for ConsoleLogStream.line_printed — mirrors terminal output."""
        self.log_text.append(line)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        if self.is_scanning:
            self.stop_scan()
        # Stop the live preview loop and release the cameras on exit
        self.stop_preview()
        self.camera_manager.stop()
        # Let any in-flight report email finish so the scan report is not lost on exit
        for thread in list(self.email_threads):
            thread.wait(90000)
        self.led_controller.cleanup()
        self.temp_sensor.cleanup()
        # Restore original stdout/stderr so post-shutdown messages go to the real terminal
        sys.stdout = self._console_stream._original
        sys.stderr = self._console_stream_err._original
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Show loading screen first. The MainWindow is NOT created here - creating it
    # during the loading screen would initialize all hardware (cameras, LED
    # controller, detector) while the loading screen is still visible, making it
    # look like two instances of the application are running at once. Instead the
    # main window is created only after the loading screen finishes.
    logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
    from src.gui.loading_screen import LoadingScreen
    loading_screen = LoadingScreen(logo_path)
    loading_screen.show()

    # Create and show the main window only after the loading screen completes
    loading_screen.loading_complete.connect(lambda: show_main_window(loading_screen))

    sys.exit(app.exec_())


def show_main_window(loading_screen):
    """Transition from loading screen to main window.

    The MainWindow is created here (not during the loading screen) so hardware
    initialization only happens once, after the loading screen closes."""
    loading_screen.close()
    window = MainWindow()
    window.show()


if __name__ == '__main__':
    main()
