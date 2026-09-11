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
                             QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QTextCursor
from typing import Optional, TextIO
import io
import os
import queue
import shutil
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import modules
from src.hardware.camera import DualCameraManager
from src.logging.logger import DetectionLogger
from src.notification.email_notifier import EmailNotifier
from src.backend.database import get_database
from src.backend import report_builder


class DetectionThread(QThread):
    frame_ready_left = pyqtSignal(np.ndarray, DetectionResult)
    frame_ready_right = pyqtSignal(np.ndarray, DetectionResult)
    # count, avg_confidence, activity, left_count, right_count
    detection_update = pyqtSignal(int, float, str, int, int)
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
        try:
            # Pre-warm the NCNN model on this thread. NCNN does thread-local
            # initialization on the first inference call (prints "Loading
            # models..."), which can segfault if it happens mid-loop. Running
            # a dummy frame first ensures the initialization is complete.
            import numpy as np
            dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.detector.detect(dummy)

            while self.running:
                cycle_started = time.perf_counter()

                # Check which cameras are delivering live frames. Each camera
                # is checked independently — if one is missing/disconnected,
                # detection still runs on the other. is_healthy() verifies the
                # camera is actually delivering frames (not just opened).
                left_healthy = self.camera_manager.left_camera.is_healthy()
                right_healthy = self.camera_manager.right_camera.is_healthy()
                self.camera_status.emit(left_healthy, right_healthy)
                left_frame = self.camera_manager.get_left_frame() if left_healthy else None
                right_frame = self.camera_manager.get_right_frame() if right_healthy else None

                # Run detection independently on each camera's frame. Each
                # camera's detections are emitted separately so the preview
                # threads can overlay boxes on each feed independently.
                # NOTE: NCNN is NOT thread-safe — calling detect() from two
                # threads simultaneously on the same model segfaults, so left
                # and right must be detected sequentially.
                left_count = 0
                right_count = 0
                left_confidences = []
                right_confidences = []
                if left_frame is not None:
                    detection_left = self.detector.detect(left_frame)
                    left_count = detection_left.count
                    left_confidences = detection_left.confidences
                    self.frame_ready_left.emit(left_frame, detection_left)

                if right_frame is not None:
                    detection_right = self.detector.detect(right_frame)
                    right_count = detection_right.count
                    right_confidences = detection_right.confidences
                    self.frame_ready_right.emit(right_frame, detection_right)

                # Dual-camera count: the two cameras' fields of view overlap, so
                # the same weevil is often visible in both feeds. Summing
                # left + right would double-count those weevils and inflate the
                # treatment recommendation. Reporting max(left, right) takes
                # the count from the camera with the better view of the scene,
                # avoiding double-counting while never under-reporting what the
                # better-positioned camera sees. When only one camera is live,
                # its count stands on its own. Both per-camera counts are still
                # emitted (and stored on the detection row) for the audit trail.
                if left_frame is not None and right_frame is not None:
                    total_count = max(left_count, right_count)
                    # Average confidence comes from the winning camera so it
                    # reflects the detections actually being counted; on a tie
                    # or a zero-zero reading, average both cameras together.
                    if left_count > right_count and left_count > 0:
                        confidences = left_confidences
                    elif right_count > left_count and right_count > 0:
                        confidences = right_confidences
                    else:
                        confidences = left_confidences + right_confidences
                else:
                    total_count = left_count + right_count
                    confidences = left_confidences + right_confidences

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                self.detection_update.emit(total_count, avg_confidence, "Detection",
                                           left_count, right_count)

                elapsed_ms = (time.perf_counter() - cycle_started) * 1000
                self.stats_update.emit(1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0,
                                       self.detector.avg_inference_ms)

                # Yield to the OS so the GUI thread and camera capture thread
                # get CPU time. With DETECTION_INTERVAL_MS=0, this is a minimal
                # 1ms yield — detection runs back-to-back as fast as inference
                # allows, with no artificial delay between cycles.
                remaining = self.interval_ms - elapsed_ms
                self.msleep(int(remaining) if remaining > 0 else 1)
        except Exception as e:
            print(f"DetectionThread crashed: {e}")
            import traceback
            traceback.print_exc()

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

    def __init__(self, camera_manager: DualCameraManager, interval_ms: int = 33):
        super().__init__()
        self.camera_manager = camera_manager
        # ~30 FPS for smooth live preview. The capture thread runs independently
        # in its own thread, so this only controls how often frames are emitted
        # to the GUI for display. Detection runs on a separate thread during scans.
        self.interval_ms = max(0, interval_ms)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            cycle_started = time.perf_counter()
            # Use is_healthy() (not is_running()) so a camera that is opened but
            # delivering blank/frozen frames shows the "not detected" placeholder
            # instead of a stale image. Each camera is checked independently.
            left_healthy = self.camera_manager.left_camera.is_healthy()
            right_healthy = self.camera_manager.right_camera.is_healthy()
            self.camera_status.emit(left_healthy, right_healthy)
            # Use get_frame_raw to avoid the expensive .copy() — the frame is
            # only read (never mutated) by the rendering slot, so sharing the
            # buffer is safe as long as the capture thread replaces current_frame
            # atomically (it does, under self.lock).
            left_frame = self.camera_manager.get_left_frame_raw() if left_healthy else None
            right_frame = self.camera_manager.get_right_frame_raw() if right_healthy else None
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


class VideoWriterThread(QThread):
    """Encodes frames to a video file in a background thread.

    cv2.VideoWriter.write() is CPU-intensive (H.264/mp4v encoding). Calling it
    on the GUI thread blocks the Qt event loop, so queued signals from the
    detection thread pile up faster than they can be processed - the window
    stops repainting, button clicks are ignored, and eventually Python becomes
    unresponsive. This thread drains a bounded queue and encodes asynchronously,
    dropping frames if the encoder falls behind rather than stalling the GUI.
    """
    def __init__(self, writer: cv2.VideoWriter, max_queue: int = 30):
        super().__init__()
        self.writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.running = False

    def put(self, frame: np.ndarray):
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def run(self):
        self.running = True
        while self.running or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.5)
                try:
                    self.writer.write(frame)
                except Exception:
                    pass
            except queue.Empty:
                continue

    def stop_and_release(self):
        self.running = False
        self.wait(5000)
        try:
            self.writer.release()
        except Exception:
            pass


class ConsoleLogStream(QObject):
    """Tee stdout/stderr so every print() also appears in the System Log.

    print() is called from background threads (camera capture loop, email
    thread, etc.), so a Qt signal is used to marshal the text to the GUI
    thread safely.
    """
    # NVIDIA Broadcast's DirectShow filter spams hundreds of [DSH]/[MBHB]/[UIB]
    # log lines to stderr when a virtual camera is opened. Each line would
    # emit a Qt signal to the GUI thread, flooding the event loop and making
    # the app unresponsive. Filter these out before emitting.
    _SPAM_PREFIXES = ('[DSH]', '[MBHB]', '[UIB]', '[ERR ] [DSH]',
                      '[WARN] [DSH]', '[INFO] [DSH]')

    line_printed = pyqtSignal(str)

    def __init__(self, original: TextIO):
        super().__init__()
        self._original = original
        self._buffer = ""

    def _is_spam(self, line: str) -> bool:
        return any(line.strip().startswith(p) for p in self._SPAM_PREFIXES)

    def write(self, text: str) -> int:
        self._original.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip() and not self._is_spam(line):
                self.line_printed.emit(line)
        return len(text)

    def flush(self):
        self._original.flush()
        if self._buffer.strip() and not self._is_spam(self._buffer):
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
        try:
            success = self.send_func(*self.args, **self.kwargs)
            if success:
                self.finished_with_status.emit(True, f"Email sent: {self.description}")
            else:
                self.finished_with_status.emit(
                    False, f"Email FAILED: {self.description} — check config.env credentials and network")
        except Exception as e:
            import traceback
            print(f"EmailThread exception: {e}")
            traceback.print_exc()
            self.finished_with_status.emit(False, f"Email error: {self.description} ({e})")


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
    # Emitted when the label is resized or first shown, so any owner rendering
    # a pixmap into it (e.g. the stored-image preview) can re-fit the image.
    resized = pyqtSignal()

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
        self.resized.emit()

    def showEvent(self, event):
        # The hidden tab never gets a resize until it is first shown, so place the
        # panel here too - otherwise it sits in the top-left corner over the title.
        super().showEvent(event)
        self.position_zoom_panel()
        self.resized.emit()


class MainWindow(QMainWindow):
    # Touch target sizing for the Raspberry Pi 5 touchscreen at 800x400, 1x scaling.
    TOUCH_BUTTON_SIZE = 40
    # Overlay zoom buttons are smaller than the main controls but still touchable.
    ZOOM_BUTTON_SIZE = 32
    # Fixed width of the compact info column. Narrow so the camera feed gets the
    # dominant share of the 800px-wide screen.
    INFO_PANEL_WIDTH = 200
    # Table/list rows in the Detection Logs tab are selected by finger.
    TOUCH_ROW_HEIGHT = 36
    # Scrollbars are dragged by finger, so they are wider than the desktop default.
    TOUCH_SCROLLBAR_SIZE = 18
    # Maximum lines kept in the System Log.
    MAX_LOG_LINES = 500
    # Group boxes: minimal margins to save vertical space on the 400px-tall screen.
    COMPACT_GROUP_STYLE = (
        "QGroupBox { font-weight: bold; font-size: 9px; border: 1px solid #ccc; "
        "border-radius: 3px; margin-top: 5px; padding-top: 1px; } "
        "QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 2px; }")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anilag - Rice Weevil Detection and Control System")
        # Target display is the Raspberry Pi 5 touchscreen at 800x400, 1x scaling.
        # The systemd service hides the taskbar (lxpanel) before launching, so
        # we can use the FULL screen geometry instead of availableGeometry.
        # Fallback to availableGeometry if the taskbar is still visible.
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        full = screen.geometry()
        available = screen.availableGeometry()
        # If the taskbar is hidden, available == full. Use whichever is bigger.
        geo = full if (full.width() * full.height()) >= (available.width() * available.height()) else available
        self.setGeometry(geo)
        self.setMinimumSize(640, 360)
        # showFullScreen gives us the entire screen with no title bar or borders.
        # showMaximized is the fallback if the WM doesn't support fullscreen.
        self.showFullScreen()

        # Tee stdout/stderr into the System Log so every print() from any
        # module (camera, detector, email) appears in the
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
            width=int(os.getenv('CAMERA_WIDTH', '1920')),
            height=int(os.getenv('CAMERA_HEIGHT', '1080')),
            fps=int(os.getenv('CAMERA_FPS', '30'))
        )
        
        self.detector = YOLOv11Detector(
            model_path=os.getenv('MODEL_PATH', 'models/sitophilus_oryzae_v2-3_best.pt'),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
            iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.7'))
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
        self.detection_data = []  # List of tuples: (timestamp, count, recommendation)
        
        # Track last log time for 1-minute interval logging
        self.last_log_time = None
        
        # Recording infrastructure
        self.video_writer_left = None
        self.video_writer_right = None
        self.video_writer_thread_left = None
        self.video_writer_thread_right = None
        self.current_scan_folder = None
        self.current_scan_id = None
        _default_scans_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'previous_scans')
        _scan_storage = os.getenv('SCAN_STORAGE_PATH', '').strip()
        self.previous_scans_dir = os.path.join(_scan_storage, 'previous_scans') if _scan_storage else _default_scans_dir
        os.makedirs(self.previous_scans_dir, exist_ok=True)
        
        # Scan metadata
        self.scan_start_time = None
        self.scan_max_count = 0
        
        # Image capture settings
        self.high_weevil_threshold = int(os.getenv('HIGH_WEEVIL_THRESHOLD', '5'))
        self.last_image_capture_time = None
        self._last_raw_only_capture_time = None
        self.image_capture_cooldown = int(os.getenv('IMAGE_CAPTURE_COOLDOWN', '30'))  # seconds between captures
        # Also store the CLAHE-preprocessed frame (exactly what the model saw)
        # alongside each annotated capture, for detection auditing. Only takes
        # effect when the detector's CLAHE preprocessing is enabled; otherwise
        # the preprocessed frame is identical to the raw and would duplicate it.
        self.store_preprocessed_images = os.getenv('STORE_PREPROCESSED_IMAGES', 'true').lower() in ('true', '1', 'yes', 'on')
        self.latest_annotated_left = None
        self.latest_annotated_right = None
        # Cached detection results from DetectionThread, overlaid on preview frames
        self._latest_detection_left = None
        self._latest_detection_right = None
        # Cached frames from the detection thread — the EXACT frames that were
        # passed to detector.detect(). Used by capture_detection_images so the
        # stored annotated image has boxes drawn on the same frame the model
        # actually saw, not a later frame where the weevil has already moved.
        self._latest_detected_frame_left = None
        self._latest_detected_frame_right = None
        # Previous detection results for box interpolation (smooth tracking).
        # When a weevil moves between detection cycles (10 FPS), the preview
        # thread (30 FPS) linearly interpolates box positions so the bounding
        # box appears to smoothly follow the weevil instead of jumping.
        self._prev_detection_left = None
        self._prev_detection_right = None
        self._detection_timestamp_left = 0.0
        self._detection_timestamp_right = 0.0
        # Actual measured time between detection cycles (seconds). The
        # configured DETECTION_INTERVAL_MS is just a minimum sleep — the real
        # interval is dominated by inference time (~1-4s with tiling). The
        # interpolation uses this to smoothly move boxes between detection
        # cycles so they follow the moving weevil at the 30 FPS preview rate.
        self._detection_interval_s = 1.0  # updated dynamically
        self._prev_detection_timestamp_left = 0.0
        self._prev_detection_timestamp_right = 0.0

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
        # Heating Timer: a 3-minute safety timer shown in the camera feed tabs.
        # The user starts it manually to track a heat treatment cycle and is
        # warned before stopping it early because interrupting may under-heat
        # the rig.
        self.heating_timer_duration_seconds = int(os.getenv('HEATING_TIMER_SECONDS', '180'))
        self.heating_remaining_seconds = self.heating_timer_duration_seconds
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
        # Detection cycle interval - throttles YOLO inference on the Pi 5 CPU
        self.detection_interval_ms = int(os.getenv('DETECTION_INTERVAL_MS', '33'))
        # _detection_interval_s is updated dynamically in _on_detection_left/right
        # to match the actual measured time between detection cycles.
        self.scan_detection_count = 0
        self.scan_image_count = 0
        self.reports_dir = os.path.join(self.previous_scans_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup UI
        self.init_ui()
        self.initialize_components()

        # Heating Timer - fires once when the 3-minute heating cycle completes.
        self.heating_timer = QTimer()
        self.heating_timer.setSingleShot(True)
        self.heating_timer.timeout.connect(self.on_heating_timer_elapsed)
        self.heating_countdown_timer = QTimer()
        self.heating_countdown_timer.timeout.connect(self.update_heating_countdown)

        # Setup clock update timer
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  # Update every second

        # Camera hot-plug monitor: checks every 2 seconds whether a camera has
        # been disconnected (unplugged) and tries to re-open it if so. When a
        # camera is unplugged, cap.read() fails and the capture loop sets
        # signal_live=False. When the user plugs it back in, this timer detects
        # the camera is not running and re-initializes it so the feed resumes
        # smoothly without requiring a restart.
        self.camera_monitor_timer = QTimer()
        self.camera_monitor_timer.timeout.connect(self._check_camera_hotplug)
        self.camera_monitor_timer.start(2000)  # Check every 2 seconds

        # Paint both immediately so the UI never shows a blank clock
        # for the first second after startup.
        self.update_clock()
        
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
        header_widget.setStyleSheet("background-color: #f5f5f5; border-bottom: 1px solid #ddd;")
        header_widget.setFixedHeight(40)

        # Logo label - fills the header height, no box/border around it
        self.logo_label = QLabel()
        self.logo_label.setStyleSheet("background: transparent; border: none; padding: 0px;")
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
        self._logo_path = logo_path
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale to fill the header height (36px fits the 40px header with 2px margin)
            pixmap = pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(pixmap)
            self.logo_label.setMinimumSize(36, 36)
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
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2E7D32; background: transparent;")
        title_layout.addWidget(title_label)

        # Tagline label
        tagline_label = QLabel("Rice Weevil Detection and Control System")
        tagline_label.setFont(QFont("Arial", 8))
        tagline_label.setStyleSheet("color: #666; background: transparent;")
        title_layout.addWidget(tagline_label)

        header_layout.addWidget(title_container)
        
        header_layout.addStretch()
        
        # Always-visible real-time clock (updated every second from the system clock)
        self.header_clock_label = QLabel()
        self.header_clock_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.header_clock_label.setStyleSheet("color: #2E7D32; padding-right: 4px;")
        self.header_clock_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.header_clock_label)
        
        main_layout.addWidget(header_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Camera feed tabs. Both tabs share one instance of the Scan Controls,
        # Current Detection and Detection Log panels, which are
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
                font-size: 11px;
                font-weight: bold;
                border: {border};
                border-radius: 4px;
                padding: 4px;
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
        btn.setMinimumWidth(60)
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
                font-size: 16px; font-weight: bold;
                border-radius: 4px; padding: 1px;
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
        zoom_label.setFont(QFont("Arial", 8, QFont.Bold))
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setFixedSize(self.ZOOM_BUTTON_SIZE, 18)
        zoom_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 160); border-radius: 3px;")
        panel_layout.addWidget(zoom_in)
        panel_layout.addWidget(zoom_label)
        panel_layout.addWidget(zoom_out)
        # Fixed size keeps the placement maths correct before the panel is first shown.
        panel.setFixedSize(self.ZOOM_BUTTON_SIZE + 6, self.ZOOM_BUTTON_SIZE * 2 + 18 + 12)
        parent_label.zoom_panel = panel
        parent_label.position_zoom_panel()
        panel.show()
        return panel, zoom_in, zoom_out, zoom_label

    def create_feed_tab(self, side: str) -> QWidget:
        """Build one camera feed tab. The tab only owns the camera view; the
        Scan Controls, Current Detection and Detection Log panels
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
        # Placeholder text shown until the camera delivers frames. Updated by
        # update_camera_status() when cameras are detected or lost.
        camera_label.setText(f"Camera{'0' if side == 'left' else '1'} not detected")
        left_panel.addWidget(camera_label, stretch=1)

        camera_title = QLabel(f"Camera{'0' if side == 'left' else '1'}")
        camera_title.setFont(QFont("Arial", 8, QFont.Bold))
        camera_title.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 2px; border-radius: 3px;")
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

        # Scan and heating controls sit side by side in a single strip so the strip
        # stays one row tall and the camera feed keeps the rest of the tab.
        self.shared_controls_widget = QWidget()
        # Fixed height so the controls strip never expands vertically.
        # Compact for the 400px-tall Pi 5 screen.
        self.shared_controls_widget.setFixedHeight(56)
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
        # Expand to fill the Scan Controls group so the button is as wide as the
        # Heating Timer group beside it, balancing the two halves of the strip.
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        scan_layout.addWidget(self.start_button)

        scan_group.setLayout(scan_layout)
        controls_layout.addWidget(scan_group, stretch=1)

        # Heating Timer beside the scan controls. Shows an enlarged countdown,
        # a spinbox to change the duration (default 3 minutes), and a single
        # Start/Stop button. The user starts it manually to track a heat
        # treatment cycle and can stop it early but is warned first.
        heating_group = QGroupBox("Heating Timer")
        heating_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        heating_layout = QHBoxLayout()
        heating_layout.setContentsMargins(6, 4, 6, 4)
        heating_layout.setSpacing(6)

        # Minutes selector - up/down arrow buttons stacked vertically. The
        # current value is shown only on the countdown display to the right, so
        # there is no redundant number here. Default 3 minutes, adjustable 1-60.
        # Disabled while the timer runs so a mid-cycle change cannot desync the
        # countdown from the single-shot QTimer.
        arrow_style = (
            "QPushButton { background-color: #ffe0b2; color: #e65100; "
            "font-size: 12px; font-weight: bold; border: 1px solid #ffb74d; "
            "border-radius: 3px; padding: 0px; }"
            "QPushButton:disabled { background-color: #f5f5f5; color: #bbb; "
            "border: 1px solid #ddd; }"
            "QPushButton:hover { background-color: #ffb74d; }"
            "QPushButton:pressed { background-color: #ff9800; color: white; }")
        arrow_box = QVBoxLayout()
        arrow_box.setContentsMargins(0, 0, 0, 0)
        arrow_box.setSpacing(1)
        self.heating_up_button = QPushButton("\u25B2")
        self.heating_up_button.setFixedSize(28, self.TOUCH_BUTTON_SIZE // 2)
        self.heating_up_button.setStyleSheet(arrow_style)
        self.heating_up_button.clicked.connect(self.increment_heating_minutes)
        arrow_box.addWidget(self.heating_up_button)
        self.heating_down_button = QPushButton("\u25BC")
        self.heating_down_button.setFixedSize(28, self.TOUCH_BUTTON_SIZE // 2)
        self.heating_down_button.setStyleSheet(arrow_style)
        self.heating_down_button.clicked.connect(self.decrement_heating_minutes)
        arrow_box.addWidget(self.heating_down_button)
        heating_layout.addLayout(arrow_box)

        # Enlarged countdown display - big enough to read at a glance on the
        # 7-inch touchscreen from arm's length. This is the only place the
        # current duration is shown.
        self.heating_display_label = QLabel(
            f"{self.heating_timer_duration_seconds // 60:02d}:00")
        self.heating_display_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.heating_display_label.setAlignment(Qt.AlignCenter)
        self.heating_display_label.setStyleSheet(
            "padding: 4px 8px; background-color: #fff3e0; border-radius: 4px; "
            "border: 1px solid #ffb74d; color: #e65100;")
        self.heating_display_label.setMinimumWidth(80)
        heating_layout.addWidget(self.heating_display_label)

        self.heating_toggle_button = self._make_touch_button(
            "Start", "#ff9800", "#f57c00", self.toggle_heating_timer,
            pressed="#ef6c00")
        heating_layout.addWidget(self.heating_toggle_button)

        heating_group.setLayout(heating_layout)
        controls_layout.addWidget(heating_group, stretch=1)
        
        # Current detection info
        current_info_group = QGroupBox("Current Detection")
        current_info_group.setStyleSheet(self.COMPACT_GROUP_STYLE)
        current_info_layout = QVBoxLayout()
        current_info_layout.setSpacing(4)
        current_info_layout.setContentsMargins(6, 4, 6, 6)

        self.count_label = QLabel("Weevil Count: --")
        self.count_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("padding: 4px; background-color: #e8f5e9; border-radius: 4px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.count_label)

        # Confidence label stacked under the count.
        self.confidence_label = QLabel("Avg Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 9))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("padding: 3px; background-color: #e8f5e9; border-radius: 4px; border: 1px solid #c8e6c9;")
        current_info_layout.addWidget(self.confidence_label)

        self.recommendation_label = QLabel("Recommendation: --")
        self.recommendation_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setMinimumWidth(0)
        self.recommendation_label.setStyleSheet("color: #0066cc; padding: 3px; background-color: #fff3e0; border-radius: 4px; border: 1px solid #ffe0b2;")
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
        self.inference_label.setFont(QFont("Consolas", 9, QFont.Bold))
        self.inference_label.setAlignment(Qt.AlignCenter)
        self.inference_label.setStyleSheet("padding: 3px; background-color: #fff8e1; border-radius: 4px; border: 1px solid #ffecb3; color: #5d4037;")
        model_layout.addWidget(self.inference_label)

        self.rate_label = QLabel("Rate: --")
        self.rate_label.setFont(QFont("Consolas", 9, QFont.Bold))
        self.rate_label.setAlignment(Qt.AlignCenter)
        self.rate_label.setStyleSheet("padding: 3px; background-color: #fff8e1; border-radius: 4px; border: 1px solid #ffecb3; color: #5d4037;")
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
        self.log_text.setMinimumHeight(40)
        self.log_text.setMaximumHeight(100)
        self.log_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.log_text.setStyleSheet("font-family: Consolas, monospace; font-size: 9px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;")
        self._make_touch_scrollable(self.log_text)
        log_layout.addWidget(self.log_text)

        # Buffer incoming log lines and flush them on a timer so a burst of
        # print() calls from background threads (e.g. a failing camera polling
        # at 10 Hz) does not flood the GUI event loop with one QTextEdit.append()
        # per line. append() is O(n) in the document size, so thousands of
        # separate appends make the UI unresponsive. The flush timer coalesces
        # the whole burst into a single append + reflow, and _trim_log_text
        # caps the document so append() stays cheap over time.
        self._log_buffer = []
        self._log_flush_timer = QTimer()
        self._log_flush_timer.setInterval(100)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._log_flush_timer.start()
        
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
                font-size: 10px;
            }}
            QTableWidget::item {{
                padding: 6px 4px;
                border-bottom: 1px solid #eee;
            }}
            QTableWidget::item:selected {{
                background-color: #1976D2;
                color: white;
            }}
            QHeaderView::section {{
                background-color: #f5f5f5;
                padding: 6px 4px;
                border: 1px solid #ddd;
                font-weight: bold;
                font-size: 10px;
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
        self.scan_table.setColumnCount(6)
        self.scan_table.setHorizontalHeaderLabels(
            ["Scan ID", "Start Time", "End Time", "Scan Time", "Max Count", "Images"])
        self.scan_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.scan_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.scan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.scan_table.setSelectionMode(QTableWidget.SingleSelection)
        self.scan_table.setStyleSheet(table_style)
        self.scan_table.setMinimumWidth(300)
        self._make_touch_scrollable(self.scan_table)
        # Tall rows and no row numbers: the vertical header is dead space on a
        # touchscreen and the scan id already identifies the row.
        self.scan_table.verticalHeader().setDefaultSectionSize(self.TOUCH_ROW_HEIGHT)
        self.scan_table.verticalHeader().setVisible(False)
        self.scan_table.itemSelectionChanged.connect(self.on_scan_selected)
        scans_layout.addWidget(self.scan_table)
        
        scans_group.setLayout(scans_layout)
        splitter.addWidget(scans_group)
        
        # Right: images stored for the selected scan
        detail_group = QGroupBox("Stored Images")
        detail_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(5)
        
        self.image_preview_label = QLabel("Select a scan to preview its stored images")
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setMinimumHeight(120)
        self.image_preview_label.setStyleSheet("border: 1px solid #333; background-color: #000; color: #999; border-radius: 3px; font-size: 9px;")
        # The inline preview only fits the selected image to the panel. Zooming
        # and panning happen in the full-image pop-up (View Full Image), so no
        # zoom overlay or drag-to-pan handling is needed here.
        detail_layout.addWidget(self.image_preview_label, stretch=1)

        self.image_list = QListWidget()
        # Show several finger-sized rows so the stored image list is clearly
        # visible and scrollable. The list takes a healthy share of the pane
        # alongside the preview above it.
        self.image_list.setMinimumHeight(self.TOUCH_ROW_HEIGHT * 4)
        self.image_list.setStyleSheet(f"""
            QListWidget {{
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 2px solid #1976D2;
                border-radius: 4px;
                background-color: #fafafa;
            }}
            QListWidget::item {{
                min-height: {self.TOUCH_ROW_HEIGHT}px;
                padding: 6px 8px;
                border-bottom: 1px solid #e0e0e0;
            }}
            QListWidget::item:selected {{ background-color: #1976D2; color: white; }}
            QScrollBar:vertical {{
                background: #e3f2fd;
                width: {self.TOUCH_SCROLLBAR_SIZE}px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #1976D2;
                border-radius: {self.TOUCH_SCROLLBAR_SIZE // 2}px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:pressed {{ background: #0D47A1; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        self._make_touch_scrollable(self.image_list)
        self.image_list.currentRowChanged.connect(self.on_history_image_selected)
        # Double-clicking a stored image (left or right capture) opens it full
        # size in a closeable pop-up window so it can be inspected in detail.
        self.image_list.itemDoubleClicked.connect(lambda _item: self._open_image_popup())

        # Image list and "View Full Image" button sit side-by-side so they take
        # one row instead of two, leaving more room for the preview and log.
        image_row = QHBoxLayout()
        image_row.setSpacing(4)
        image_row.addWidget(self.image_list, stretch=1)

        self.view_image_button = QPushButton("View Full Image")
        self.view_image_button.setMinimumHeight(self.TOUCH_ROW_HEIGHT + 6)
        self.view_image_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.view_image_button.setStyleSheet("""
            QPushButton {
                background-color: #00897B; color: white;
                font-size: 10px; font-weight: bold;
                border-radius: 4px; padding: 4px 8px;
            }
            QPushButton:hover { background-color: #00695C; }
            QPushButton:pressed { background-color: #004D40; }
            QPushButton:disabled { background-color: #bdbdbd; }
        """)
        self.view_image_button.clicked.connect(self._open_image_popup)
        image_row.addWidget(self.view_image_button, stretch=0)
        detail_layout.addLayout(image_row, stretch=0)

        detail_group.setLayout(detail_layout)
        splitter.addWidget(detail_group)
        # Stored Scans and Stored Images share the tab width equally.
        # setSizes gives the initial split; the stretch factors (1, 1) keep
        # them equal as the window is resized.
        splitter.setSizes([400, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
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
                    font-size: 11px;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 4px;
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
        preview_fps = int(os.getenv('PREVIEW_FPS', '30'))
        preview_interval_ms = max(1, int(1000 / preview_fps))
        self.preview_thread = PreviewThread(self.camera_manager, interval_ms=preview_interval_ms)
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
        """Render a raw camera frame for live preview at the camera FPS.

        During a scan, overlay the real detection bounding boxes from the
        detection thread. Detection runs slower than the camera (each cycle
        takes ~1-4s with tiling), so between detection cycles the boxes are
        linearly interpolated from the previous to the current detection so
        they smoothly follow the moving weevil instead of staying stuck at
        the old position.
        """
        if self.is_scanning and self._latest_detection_left is not None:
            # Interpolate box positions between the previous and current
            # detection so the bounding box tracks the moving weevil at the
            # full 30 FPS preview rate, not just every detection cycle.
            interp = self._interpolate_detection(
                self._prev_detection_left,
                self._latest_detection_left,
                self._detection_timestamp_left,
                self._detection_interval_s,
            )
            annotated = self.detector.draw_detections(frame, interp)
            self.latest_annotated_left = annotated
            if self.video_writer_thread_left:
                self.video_writer_thread_left.put(annotated)
            self._render_frame_to_label(annotated, self.left_camera_label, self.left_zoom,
                                        self.left_pan_x, self.left_pan_y)
        else:
            self.latest_annotated_left = frame
            self._render_frame_to_label(frame, self.left_camera_label, self.left_zoom,
                                        self.left_pan_x, self.left_pan_y)

    def update_right_preview(self, frame: np.ndarray):
        """Render a raw camera frame for live preview (right camera)."""
        if self.is_scanning and self._latest_detection_right is not None:
            interp = self._interpolate_detection(
                self._prev_detection_right,
                self._latest_detection_right,
                self._detection_timestamp_right,
                self._detection_interval_s,
            )
            annotated = self.detector.draw_detections(frame, interp)
            self.latest_annotated_right = annotated
            if self.video_writer_thread_right:
                self.video_writer_thread_right.put(annotated)
            self._render_frame_to_label(annotated, self.right_camera_label, self.right_zoom,
                                        self.right_pan_x, self.right_pan_y)
        else:
            self.latest_annotated_right = frame
            self._render_frame_to_label(frame, self.right_camera_label, self.right_zoom,
                                        self.right_pan_x, self.right_pan_y)

    def _on_detection_left(self, frame: np.ndarray, detection: DetectionResult):
        """Cache detection results from DetectionThread.

        The preview thread overlays these cached results on every 30 FPS frame
        for smooth real-time display and recording. Box positions are interpolated
        between detection cycles so bounding boxes smoothly follow moving weevils.
        """
        if not self.is_scanning:
            return
        now = time.perf_counter()
        # Measure the actual interval between detection cycles so the
        # interpolation knows how long to spread the box movement over.
        if self._prev_detection_timestamp_left > 0:
            self._detection_interval_s = max(0.1, now - self._prev_detection_timestamp_left)
        self._prev_detection_timestamp_left = self._detection_timestamp_left if self._detection_timestamp_left > 0 else now
        self._prev_detection_left = self._latest_detection_left
        self._latest_detection_left = detection
        self._latest_detected_frame_left = frame
        self._detection_timestamp_left = now

    def _on_detection_right(self, frame: np.ndarray, detection: DetectionResult):
        """Cache detection results from DetectionThread (right camera)."""
        if not self.is_scanning:
            return
        now = time.perf_counter()
        if self._prev_detection_timestamp_right > 0:
            self._detection_interval_s = max(0.1, now - self._prev_detection_timestamp_right)
        self._prev_detection_timestamp_right = self._detection_timestamp_right if self._detection_timestamp_right > 0 else now
        self._prev_detection_right = self._latest_detection_right
        self._latest_detection_right = detection
        self._latest_detected_frame_right = frame
        self._detection_timestamp_right = now

    def _interpolate_detection(self, prev: DetectionResult, curr: DetectionResult,
                               curr_time: float, interval: float) -> DetectionResult:
        """Extrapolate box positions forward from the current detection.

        The detection cycle takes ~1-4s (2x2 tiling), but the preview runs at
        30 FPS. To make boxes follow the moving weevil, we extrapolate each
        box forward from its current position in the direction of recent
        movement (curr - prev).

        To prevent ghost detections:
        - Extrapolation is capped at 30% of the movement vector (t_max=0.3)
          and 20px max displacement, so the box never overshoots far from the
          last detected position.
        - Match distance is tight (80px) so unrelated boxes don't get matched.
        - Boxes that disappear (in prev but not curr) are dropped immediately.
        - New unmatched boxes are shown at their detected position (no
          extrapolation) — if they're false positives, they vanish on the
          next detection cycle.
        """
        if prev is None or curr is None:
            return curr
        if not curr.boxes:
            return curr
        if not prev.boxes:
            return curr

        elapsed = time.perf_counter() - curr_time
        # Cap extrapolation at 0.3 — the weevil may have changed direction
        # during the detection cycle, so projecting too far forward creates
        # ghost boxes ahead of where the weevil actually is. A small forward
        # nudge keeps the box close to the last confirmed detection.
        t = min(elapsed / interval, 0.3) if interval > 0 else 0.0
        if t <= 0.0:
            return curr

        MAX_EXTRAP_PX = 40  # max pixels a box can move from its detected position

        # Match each current box to the nearest previous box by centroid distance.
        used_prev = set()
        extrap_boxes = []
        extrap_confs = []
        extrap_cls = []
        for i, (cx1, cy1, cx2, cy2) in enumerate(curr.boxes):
            ccx = (cx1 + cx2) / 2
            ccy = (cy1 + cy2) / 2
            best_j = -1
            best_dist = float('inf')
            for j, (px1, py1, px2, py2) in enumerate(prev.boxes):
                if j in used_prev:
                    continue
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                dist = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0 and best_dist < 80:  # tight match distance
                used_prev.add(best_j)
                px1, py1, px2, py2 = prev.boxes[best_j]
                # Forward extrapolation: move box in the direction of recent
                # movement (curr - prev), capped to MAX_EXTRAP_PX.
                dx1 = cx1 - px1
                dy1 = cy1 - py1
                dx2 = cx2 - px2
                dy2 = cy2 - py2
                ex1 = dx1 * t
                ey1 = dy1 * t
                ex2 = dx2 * t
                ey2 = dy2 * t
                # Cap each displacement to MAX_EXTRAP_PX
                mag1 = (ex1 ** 2 + ey1 ** 2) ** 0.5
                if mag1 > MAX_EXTRAP_PX:
                    scale = MAX_EXTRAP_PX / mag1
                    ex1 *= scale
                    ey1 *= scale
                mag2 = (ex2 ** 2 + ey2 ** 2) ** 0.5
                if mag2 > MAX_EXTRAP_PX:
                    scale = MAX_EXTRAP_PX / mag2
                    ex2 *= scale
                    ey2 *= scale
                ix1 = int(cx1 + ex1)
                iy1 = int(cy1 + ey1)
                ix2 = int(cx2 + ex2)
                iy2 = int(cy2 + ey2)
                extrap_boxes.append([ix1, iy1, ix2, iy2])
                extrap_confs.append(curr.confidences[i])
                extrap_cls.append(curr.class_ids[i])
            else:
                # No match — new weevil appeared, use current box as-is (no
                # extrapolation since we don't know its velocity yet).
                extrap_boxes.append([cx1, cy1, cx2, cy2])
                extrap_confs.append(curr.confidences[i])
                extrap_cls.append(curr.class_ids[i])
        return DetectionResult(extrap_boxes, extrap_confs, extrap_cls, curr.class_names,
                                curr.inference_ms)

    def update_model_display(self):
        """Report which YOLOv11n weights and backend are loaded to the Detection Log.

        The Detection Model panel itself only shows the live Inference and Rate
        figures, so the static details go to the log instead of the side panel.
        """
        info = self.detector.get_model_info()
        status = "loaded" if info['initialized'] else "NOT LOADED"
        classes = ', '.join(str(c) for c in info['classes']) or 'none'
        metrics = info['metrics']
        if self.detection_interval_ms > 0:
            rate_str = f"Detection capped at {1000.0 / self.detection_interval_ms:.0f} FPS (actual rate is set by inference time)"
        else:
            rate_str = "Detection running at model max speed (no interval cap)"
        self.log_message(
            f"Model: {info['architecture']} - {info['model_name']} ({status})   |   "
            f"Backend: {info['backend']}   |   Device: {info['device']}   |   "
            f"Input: {info['imgsz']}px (trained {info['trained_imgsz']}px)   |   "
            f"Classes: {classes}   |   Confidence >= {info['confidence_threshold']}   |   "
            f"IoU {info['iou_threshold']}   |   "
            f"Tiles: {info['tile_grid']} ({info['tile_passes']} passes/frame)   |   "
            f"{rate_str}   |   "
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
        try:
            if self.is_scanning:
                self.stop_scan()
            else:
                self.start_scan()
        except Exception as e:
            import traceback
            error_msg = f"Error during scan toggle: {e}"
            self.log_message(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Scan Error", f"An error occurred:\n{e}\n\nThe application will continue running.")
            # Ensure UI state is consistent after an error
            self.is_scanning = False
            self.start_button.setText("Start Scan")
            self.start_button.setStyleSheet(
                self._touch_button_style("#4CAF50", "#45a049", pressed="#3d8b40"))
            self.status_label.setText("Ready")

    @staticmethod
    def _create_video_writer(path: str, fps: int, frame_size: tuple) -> Optional[cv2.VideoWriter]:
        """Create a VideoWriter that works without external DLLs on Windows.

        OpenCV's FFMPEG backend needs openh264.dll for H.264 (avc1) and even
        mp4v in an .mp4 container. isOpened() can return True even when the
        codec failed internally, so we validate by writing a test frame and
        checking the output file is non-empty. Returns None if no codec works,
        so the caller can skip recording gracefully.
        """
        import numpy as np
        base, _ = os.path.splitext(path)
        # Skip avc1/mp4v-in-mp4 on Windows — they always need OpenH264 DLL.
        # Go straight to .avi containers which use built-in VFW codecs.
        candidates = [
            (base + '.avi', 'mp4v'),  # MPEG-4 Part 2 in .avi — no DLL needed
            (base + '.avi', 'XVID'),  # Xvid in .avi — widely supported
            (base + '.avi', 'MJPG'),  # Motion JPEG in .avi — fallback
        ]
        test_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        for full_path, codec in candidates:
            # Remove stale file from previous failed attempt
            if os.path.exists(full_path):
                os.remove(full_path)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(full_path, fourcc, fps, frame_size)
            if not writer.isOpened():
                writer.release()
                continue
            # Validate: write a test frame and check the file is non-empty.
            # isOpened() can lie when the codec partially initialized.
            try:
                writer.write(test_frame)
                writer.release()
            except Exception:
                try:
                    writer.release()
                except Exception:
                    pass
                continue
            if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
                # Re-open the validated file for appending
                writer = cv2.VideoWriter(full_path, fourcc, fps, frame_size)
                if writer.isOpened():
                    return writer
            # Clean up failed file
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
        return None

    @staticmethod
    def _video_writer_path(writer: Optional[cv2.VideoWriter], fallback: str) -> str:
        """Return the actual output path of a VideoWriter, or empty string if None."""
        if writer is None:
            return ''
        try:
            # OpenCV doesn't expose the filename directly, but the fallback
            # path's directory is correct; only the extension may differ.
            # The _create_video_writer method may have switched .mp4 → .avi.
            base = os.path.splitext(fallback)[0]
            for ext in ('.mp4', '.avi'):
                candidate = base + ext
                if os.path.exists(candidate):
                    return candidate
        except Exception:
            pass
        return fallback

    def start_scan(self):
        # Keep the preview thread running for smooth 30 FPS display during scans.
        # The DetectionThread runs detection independently and only emits detection
        # results (not frames) — the preview slot overlays the latest bounding boxes.
        # This decouples display FPS from detection FPS, so even with augment=True
        # (1 detection/second) the camera feed stays smooth.
        try:
            self._start_scan_impl()
        except Exception as e:
            import traceback
            error_msg = f"Error starting scan: {e}"
            self.log_message(error_msg)
            traceback.print_exc()
            QMessageBox.critical(self, "Scan Error",
                f"Could not start the scan:\n{e}\n\nThe application will continue running.")
            # Reset state so the UI is not stuck in a half-started scan
            self.is_scanning = False
            self.start_button.setText("Start Scan")
            self.start_button.setStyleSheet(
                self._touch_button_style("#4CAF50", "#45a049", pressed="#3d8b40"))
            self.status_label.setText("Ready")
            try:
                self.start_preview()
            except Exception:
                pass

    def _start_scan_impl(self):
        # --- Step 1: Ensure cameras are running ---
        try:
            if not self.camera_manager.is_running():
                if not self.camera_manager.start():
                    self.log_message("Failed to start cameras - no capture device could be opened")
                    return
        except Exception as e:
            self.log_message(f"Camera start error: {e}")
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "Camera Error", f"Could not start cameras:\n{e}")
            return

        # Check if each camera is delivering live frames. Each camera is
        # independent — the scan proceeds as long as at least one is live.
        left_ok = self.camera_manager.left_camera.is_healthy()
        right_ok = self.camera_manager.right_camera.is_healthy()

        if not left_ok and not right_ok:
            self.log_message("Scan cancelled - no camera device could be opened")
            self.update_camera_status(False, False)
            QMessageBox.warning(
                self, "No Camera",
                "No camera device could be opened, so the scan was not started.")
            self.start_preview()
            return

        if left_ok and right_ok:
            self.log_message("Both cameras opened")
        else:
            self.log_message(f"Scanning with the "
                             f"{'left' if left_ok else 'right'} camera only")
        self.update_camera_status(left_ok, right_ok)

        # --- Step 2: Create scan folder and database record ---
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_scan_folder = os.path.join(self.previous_scans_dir, f"scan_{timestamp}")
            os.makedirs(self.current_scan_folder, exist_ok=True)
            self.scan_images_dir = os.path.join(self.current_scan_folder, "detected_images")
            os.makedirs(self.scan_images_dir, exist_ok=True)
            self.current_scan_id = f"scan_{timestamp}"
        except Exception as e:
            self.log_message(f"Scan folder creation error: {e}")
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "Storage Error", f"Could not create scan folder:\n{e}")
            return

        # --- Step 3: Initialize video writers (non-fatal if they fail) ---
        width = int(os.getenv('CAMERA_WIDTH', '1920'))
        height = int(os.getenv('CAMERA_HEIGHT', '1080'))
        fps = int(os.getenv('CAMERA_FPS', '30'))
        left_video_path = os.path.join(self.current_scan_folder, "left_camera.mp4")
        right_video_path = os.path.join(self.current_scan_folder, "right_camera.mp4")

        try:
            self.video_writer_left = self._create_video_writer(left_video_path, fps, (width, height))
            self.video_writer_right = self._create_video_writer(right_video_path, fps, (width, height)) if right_ok else None
            left_video_path = self._video_writer_path(self.video_writer_left, left_video_path)
            right_video_path = self._video_writer_path(self.video_writer_right, right_video_path)
            self.video_writer_thread_left = VideoWriterThread(self.video_writer_left) if self.video_writer_left else None
            self.video_writer_thread_right = VideoWriterThread(self.video_writer_right) if self.video_writer_right else None
            if self.video_writer_thread_left:
                self.video_writer_thread_left.start()
            if self.video_writer_thread_right:
                self.video_writer_thread_right.start()
            if not self.video_writer_left and not self.video_writer_right:
                self.log_message("WARNING: video recording disabled (no compatible codec found)")
        except Exception as e:
            self.log_message(f"Video writer init error (recording disabled): {e}")
            import traceback; traceback.print_exc()
            self.video_writer_left = None
            self.video_writer_right = None
            self.video_writer_thread_left = None
            self.video_writer_thread_right = None

        # --- Step 4: Reset scan metadata and create DB record ---
        try:
            self.scan_start_time = datetime.now()
            self.scan_max_count = 0
            self.scan_detection_count = 0
            self.scan_image_count = 0
            self.latest_annotated_left = None
            self.latest_annotated_right = None
            self.left_video_path = left_video_path
            self.right_video_path = right_video_path
            start_time_str = self.scan_start_time.strftime("%Y-%m-%d %H:%M:%S")
            self.db.create_scan(self.current_scan_id, start_time_str, left_video_path, right_video_path)
        except Exception as e:
            self.log_message(f"Database error: {e}")
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "Database Error", f"Could not create scan record:\n{e}")
            return

        # --- Step 5: Start detection thread ---
        try:
            self.detection_thread = DetectionThread(self.camera_manager, self.detector,
                                                    interval_ms=self.detection_interval_ms)
            self.detection_thread.frame_ready_left.connect(self._on_detection_left)
            self.detection_thread.frame_ready_right.connect(self._on_detection_right)
            self.detection_thread.detection_update.connect(self.update_detection)
            self.detection_thread.stats_update.connect(self.update_performance_stats)
            self.detection_thread.camera_status.connect(self.update_camera_status)
            self.detection_thread.start()
        except Exception as e:
            self.log_message(f"Detection thread start error: {e}")
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "Detection Error", f"Could not start detection:\n{e}")
            return
        
        # Reset image/log tracking so the previous scan's timings do not leak in
        self.last_image_capture_time = None
        self.last_log_time = None
        self.last_detection_log_time = None
        
        # Ensure the preview thread is running for smooth 30 FPS display.
        # DetectionThread handles only detection; preview handles display.
        if self.preview_thread is None:
            self.start_preview()
        
        self.is_scanning = True
        self.start_button.setText("Stop Scan")
        self.start_button.setStyleSheet(
            self._touch_button_style("#f44336", "#d32f2f", pressed="#b71c1c"))
        # Reset the recommendation; it will only be shown after the scan ends.
        self.recommendation_label.setText("Recommendation: --")
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
        scan_id = self.current_scan_id
        try:
            self.scan_timer.stop()
            self.scan_countdown_timer.stop()

            if self.detection_thread:
                self.detection_thread.stop()
                self.detection_thread = None

            # Stop recording and save metadata
            if self.video_writer_thread_left:
                self.video_writer_thread_left.stop_and_release()
                self.video_writer_thread_left = None
            elif self.video_writer_left:
                try:
                    self.video_writer_left.release()
                except Exception:
                    pass
            self.video_writer_left = None
            if self.video_writer_thread_right:
                self.video_writer_thread_right.stop_and_release()
                self.video_writer_thread_right = None
            elif self.video_writer_right:
                try:
                    self.video_writer_right.release()
                except Exception:
                    pass
            self.video_writer_right = None

            # Persist the scan metadata; the detection log and images are already in the DB
            if self.current_scan_folder and scan_id:
                self.save_scan_metadata()

            self.is_scanning = False
            self.start_button.setText("Start Scan")
            self.start_button.setStyleSheet(
                self._touch_button_style("#4CAF50", "#45a049", pressed="#3d8b40"))
            # Reset the per-frame detection display. The final collective
            # recommendation based on the scan's max weevil count is shown below.
            # "Activate Mix" / "Unload Rice" are post-scan decisions, not per-frame
            # readings.
            self.count_label.setText("Weevil Count: --")
            self.confidence_label.setText("Avg Confidence: --")
            final_rec = self.logger.generate_recommendation(
                self.scan_max_count, self.is_after_mixing, is_final=True)
            self.recommendation_label.setText(f"Recommendation: {final_rec}")
            self.status_label.setText("Ready")
            self.latest_annotated_left = None
            self.latest_annotated_right = None
            self._latest_detection_left = None
            self._latest_detection_right = None
            self._latest_detected_frame_left = None
            self._latest_detected_frame_right = None
            self.log_message("Scan stopped and saved")
        except Exception as e:
            self.is_scanning = False
            self.start_button.setText("Start Scan")
            self.start_button.setStyleSheet(
                self._touch_button_style("#4CAF50", "#45a049", pressed="#3d8b40"))
            self.status_label.setText("Ready")
            self.log_message(f"Error during scan stop: {e}")
            import traceback
            traceback.print_exc()

        # Keep the cameras open and restart the preview loop so the feeds stay
        # live after the scan ends. The cameras are only stopped on app exit.
        try:
            self.start_preview()
        except Exception as e:
            self.log_message(f"Error restarting preview: {e}")

        # Always attempt to email the report, even if cleanup above had errors
        if scan_id:
            try:
                self.email_scan_report(scan_id)
                self.refresh_scan_history()
            except Exception as e:
                self.log_message(f"Error emailing scan report: {e}")

    def email_scan_report(self, scan_id: str) -> Optional[str]:
        """Build the report archive from the database and email it automatically.

        Called when a scan stops (either the user clicked Stop Scan or the 3-minute
        protocol timer elapsed). Everything in the archive - the detection log CSV
        and the captured rice weevil images - is read back out of SQLite.
        """
        self.log_message(f"Preparing scan report for {scan_id}...")
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
            import traceback
            traceback.print_exc()
            return None

        if not archive_path:
            self.log_message(f"Could not build archive: {info.get('error')}")
            return None

        self.log_message(
            f"Scan archive built: {os.path.basename(archive_path)} "
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

        self.log_message(f"Sending scan report email to {self.email_notifier.recipient_email}...")
        self.send_email_async(f"Scan Report {scan_id}", self.email_notifier.send_scan_report,
                              scan_id, archive_path, summary, report_id=report_id)
        return archive_path

    def send_email_async(self, description: str, send_func, *args, report_id: Optional[int] = None, **kwargs):
        if not self.email_notifier.enabled:
            self.log_message(f"Email skipped ({description}): email disabled or credentials missing in config.env")
            return
        self.log_message(f"Sending email: {description}...")
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
        """Load the selected scan's stored images out of the database."""
        scan_id = self.selected_scan_id()
        self.image_list.clear()
        self.image_preview_label.setPixmap(QPixmap())
        self.image_preview_label.setText("Select a scan to preview its stored images")
        if not scan_id:
            return
        
        try:
            images = self.db.get_scan_images(scan_id, include_blob=False)
        except Exception as e:
            self.log_message(f"Error loading scan {scan_id}: {e}")
            return
        
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
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("Image data missing from database")
            return

        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(blob), 'JPG'):
            self.image_preview_label.setPixmap(QPixmap())
            self.image_preview_label.setText("Could not decode stored image")
            return
        # Fit the stored image to the preview panel. Zooming/panning for close
        # inspection happens in the full-image pop-up (View Full Image).
        self.image_preview_label.setPixmap(
            pixmap.scaled(self.image_preview_label.size(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation))

    def _open_image_popup(self):
        """Open the currently selected stored image in a closeable pop-up window.

        The image is fitted to the screen at 1.0x (no scrollbars) and can be
        zoomed in with the + / - overlay and panned by dragging, exactly like
        the inline stored-image preview. A Close button dismisses it. Works for
        any stored image, including the left and right camera captures and the
        preprocessed (CLAHE) variants.
        """
        row = self.image_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "View Image", "Select a stored image first.")
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
            QMessageBox.warning(self, "View Image", "Image data missing from database.")
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(bytes(blob), 'JPG'):
            QMessageBox.warning(self, "View Image", "Could not decode stored image.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Stored Image - {item.text().split(' (')[0]}")
        # Size the pop-up to a large fraction of the window so the image has
        # room to be inspected, but leave it resizable for the operator.
        dialog.resize(int(self.width() * 0.9), int(self.height() * 0.9))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # CameraLabel supports drag-to-pan and emits resized; we render the
        # pixmap into it with the same crop/scale zoom approach as the inline
        # stored-image preview, so the pop-up is fitted (not scrolled) at 1.0x
        # and zoomable up to the configured max.
        image_label = CameraLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 1px solid #333; background-color: #000; color: #999; border-radius: 3px; font-size: 9px;")
        image_label.setMinimumSize(320, 240)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(image_label, stretch=1)

        # Per-popup zoom/pan state. Kept local so multiple popups (if opened)
        # do not share state with each other or with the inline preview.
        state = {'pixmap': pixmap, 'zoom': 1.0, 'pan_x': 0.0, 'pan_y': 0.0}

        def render():
            pm = state['pixmap']
            if pm is None:
                return
            target = image_label.contentsRect().size()
            if target.width() <= 0 or target.height() <= 0:
                return
            zoom = state['zoom']
            if zoom > 1.0:
                w = pm.width()
                h = pm.height()
                crop_w = int(w / zoom)
                crop_h = int(h / zoom)
                cx = w / 2 + state['pan_x']
                cy = h / 2 + state['pan_y']
                x0 = int(max(0, min(w - crop_w, cx - crop_w / 2)))
                y0 = int(max(0, min(h - crop_h, cy - crop_h / 2)))
                cropped = pm.copy(x0, y0, crop_w, crop_h)
                scaled = cropped.scaled(target, Qt.KeepAspectRatioByExpanding,
                                        Qt.SmoothTransformation)
            else:
                scaled = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if scaled.width() > target.width() or scaled.height() > target.height():
                x0 = (scaled.width() - target.width()) // 2
                y0 = (scaled.height() - target.height()) // 2
                scaled = scaled.copy(x0, y0, target.width(), target.height())
            image_label.setPixmap(scaled)

        def adjust_zoom(direction):
            new = round(state['zoom'] + direction * self.zoom_step, 2)
            new = max(self.zoom_min, min(self.zoom_max, new))
            state['zoom'] = new
            state['pan_x'] = 0.0
            state['pan_y'] = 0.0
            zoom_label.setText(f"{new:.1f}x")
            render()

        def pan(delta_x, delta_y):
            zoom = state['zoom']
            if zoom <= 1.0 or state['pixmap'] is None:
                return
            pm = state['pixmap']
            img_dx = delta_x / zoom
            img_dy = delta_y / zoom
            new_x = state['pan_x'] - img_dx
            new_y = state['pan_y'] - img_dy
            w = pm.width()
            h = pm.height()
            crop_w = w / zoom
            crop_h = h / zoom
            max_x = (w - crop_w) / 2
            max_y = (h - crop_h) / 2
            state['pan_x'] = max(-max_x, min(max_x, new_x))
            state['pan_y'] = max(-max_y, min(max_y, new_y))
            render()

        image_label.pan_requested.connect(pan)
        image_label.resized.connect(render)

        # Zoom overlay (+ / 1.0x / -) pinned inside the popup label, built
        # inline so it drives the popup's local state rather than the shared
        # stored-image preview state.
        panel = QWidget(image_label)
        panel.setObjectName("zoomPanel")
        panel.setStyleSheet("#zoomPanel { background-color: rgba(0, 0, 0, 110); border-radius: 8px; }")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(4)
        zoom_in = self._make_zoom_button("+", lambda: adjust_zoom(1))
        zoom_out = self._make_zoom_button("-", lambda: adjust_zoom(-1))
        zoom_label = QLabel("1.0x")
        zoom_label.setFont(QFont("Arial", 8, QFont.Bold))
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setFixedSize(self.ZOOM_BUTTON_SIZE, 18)
        zoom_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 160); border-radius: 3px;")
        panel_layout.addWidget(zoom_in)
        panel_layout.addWidget(zoom_label)
        panel_layout.addWidget(zoom_out)
        panel.setFixedSize(self.ZOOM_BUTTON_SIZE + 6, self.ZOOM_BUTTON_SIZE * 2 + 18 + 12)
        image_label.zoom_panel = panel
        image_label.position_zoom_panel()
        panel.show()

        close_button = QPushButton("Close")
        close_button.setMinimumHeight(self.TOUCH_BUTTON_SIZE)
        close_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        close_button.setStyleSheet(
            "QPushButton { background-color: #455A64; color: white; "
            "font-size: 12px; font-weight: bold; border-radius: 4px; padding: 4px; }"
            "QPushButton:hover { background-color: #37474F; }")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, stretch=0)

        # Render once the dialog is shown so the label has a real size to fit
        # into; without this the first render sees a 0x0 contentsRect.
        dialog.show()
        render()

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
            "videos": {
                "left_camera": os.path.basename(self.left_video_path) if getattr(self, 'left_video_path', None) else None,
                "right_camera": os.path.basename(self.right_video_path) if getattr(self, 'right_video_path', None) else None
            }
        }
        
        # Save to database (metadata_json makes the scan row self-describing)
        self.db.update_scan(
            self.current_scan_id,
            end_time_str,
            self.scan_max_count,
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

    def toggle_heating_timer(self):
        """Start or stop the heating timer depending on its current state."""
        if self.heating_timer.isActive():
            self.stop_heating_timer()
        else:
            self.start_heating_timer()

    def increment_heating_minutes(self):
        """Increase the heating timer duration by 1 minute (max 60)."""
        minutes = min(60, self.heating_timer_duration_seconds // 60 + 1)
        self.heating_timer_duration_seconds = minutes * 60
        self.heating_remaining_seconds = self.heating_timer_duration_seconds
        self._update_heating_display()

    def decrement_heating_minutes(self):
        """Decrease the heating timer duration by 1 minute (min 1)."""
        minutes = max(1, self.heating_timer_duration_seconds // 60 - 1)
        self.heating_timer_duration_seconds = minutes * 60
        self.heating_remaining_seconds = self.heating_timer_duration_seconds
        self._update_heating_display()

    def start_heating_timer(self):
        """Start the heating timer for the configured duration (default 3 min)."""
        if self.heating_timer.isActive():
            return
        self.heating_remaining_seconds = self.heating_timer_duration_seconds
        self.heating_timer.start(self.heating_timer_duration_seconds * 1000)
        self.heating_countdown_timer.start(1000)
        self.heating_toggle_button.setText("Stop")
        self.heating_toggle_button.setStyleSheet(
            self._touch_button_style("#f44336", "#d32f2f", "#b71c1c"))
        # Lock the duration selector while running so a mid-cycle change cannot
        # desync the countdown from the single-shot QTimer.
        self.heating_up_button.setEnabled(False)
        self.heating_down_button.setEnabled(False)
        self.log_message(
            f"Heating timer started "
            f"({self.heating_timer_duration_seconds // 60:02d}:00)")
        self._update_heating_display()

    def stop_heating_timer(self):
        """Stop the heating timer early, warning the user first.

        Interrupting a heating cycle may leave the rig under-heated, so a
        confirmation dialog is shown before the timer is actually stopped.
        """
        if not self.heating_timer.isActive():
            return
        reply = QMessageBox.warning(
            self, "Stop Heating Timer",
            "Stopping the heating timer early may affect the control process. "
            "User may lose track of time.\n\n"
            "Stop the timer anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.heating_timer.stop()
        self.heating_countdown_timer.stop()
        self.heating_remaining_seconds = self.heating_timer_duration_seconds
        self._reset_heating_button()
        self.heating_up_button.setEnabled(True)
        self.heating_down_button.setEnabled(True)
        self.log_message("Heating timer stopped early by user")
        self._update_heating_display()

    def on_heating_timer_elapsed(self):
        """Called when the heating cycle completes."""
        self.heating_countdown_timer.stop()
        self.heating_remaining_seconds = 0
        self._reset_heating_button()
        self.heating_up_button.setEnabled(True)
        self.heating_down_button.setEnabled(True)
        self.log_message("Heating timer completed")
        self._update_heating_display()

    def update_heating_countdown(self):
        if not self.heating_timer.isActive():
            return
        remaining_ms = self.heating_timer.remainingTime()
        remaining = max(0, remaining_ms // 1000) if remaining_ms >= 0 else 0
        self.heating_remaining_seconds = remaining
        self._update_heating_display()

    def _reset_heating_button(self):
        """Restore the Start button appearance after the timer stops."""
        self.heating_toggle_button.setText("Start")
        self.heating_toggle_button.setStyleSheet(
            self._touch_button_style("#ff9800", "#f57c00", "#ef6c00"))

    def _update_heating_display(self):
        secs = self.heating_remaining_seconds
        self.heating_display_label.setText(f"{secs // 60:02d}:{secs % 60:02d}")

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
        target = label.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return

        if zoom > 1.0:
            h, w = frame.shape[:2]
            crop_w = int(w / zoom)
            crop_h = int(h / zoom)
            cx = w / 2 + pan_x
            cy = h / 2 + pan_y
            x0 = int(max(0, min(w - crop_w, cx - crop_w / 2)))
            y0 = int(max(0, min(h - crop_h, cy - crop_h / 2)))
            # Crop in BGR (cheaper than after RGB conversion), then convert
            cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)
        else:
            # No zoom: convert the full frame directly. cv2.cvtColor returns
            # a contiguous array so no extra copy is needed.
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        # QImage wraps the numpy buffer (no copy). QPixmap.fromImage copies
        # into the GPU/pixmap cache, so the numpy buffer can be freed after.
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # Scale to fill the panel, then crop the overflow. KeepAspectRatioByExpanding
        # + FastTransformation is the fastest path (nearest-neighbor, no smoothing).
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            target, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
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
        if self.video_writer_thread_left:
            self.video_writer_thread_left.put(annotated_frame)

        # Display with current zoom level (display-only; recording keeps full frame)
        self._render_frame_to_label(annotated_frame, self.left_camera_label, self.left_zoom,
                                    self.left_pan_x, self.left_pan_y)

    def update_right_frame(self, frame: np.ndarray, detection: DetectionResult):
        # Draw detections on frame
        annotated_frame = self.detector.draw_detections(frame, detection)
        self.latest_annotated_right = annotated_frame

        # Write to video file if recording (always full frame, not zoomed)
        if self.video_writer_thread_right:
            self.video_writer_thread_right.put(annotated_frame)

        # Display with current zoom level (display-only; recording keeps full frame)
        self._render_frame_to_label(annotated_frame, self.right_camera_label, self.right_zoom,
                                    self.right_pan_x, self.right_pan_y)

    def update_camera_status(self, left_live: bool, right_live: bool):
        """Clear a feed that has no live signal so no stale picture (and no stale
        bounding boxes) is left on screen for a camera that is not delivering.
        Shows a 'Camera not detected' placeholder when the camera is not live."""
        for live, label, side in ((left_live, self.left_camera_label, 'left'),
                                  (right_live, self.right_camera_label, 'right')):
            cam_num = '0' if side == 'left' else '1'
            if live:
                continue
            label.setPixmap(QPixmap())
            label.setText(f"Camera{cam_num} not detected")
            setattr(self, f"latest_annotated_{side}", None)

    def _check_camera_hotplug(self):
        """Periodically check if a disconnected camera has been re-plugged.

        When a USB camera is unplugged, the capture loop's cap.read() fails and
        the camera stops itself. This timer detects that state and tries to
        re-open the camera. If the camera is back, it restarts the capture loop
        so the feed resumes smoothly without requiring an app restart.

        Also detects a newly plugged second camera when RIGHT_CAMERA_ID was -1
        (disabled) — the camera is re-enabled and the feed appears.
        """
        try:
            for cam, side, label in (
                (self.camera_manager.left_camera, 'left', self.left_camera_label),
                (self.camera_manager.right_camera, 'right', self.right_camera_label),
            ):
                cam_num = '0' if side == 'left' else '1'
                # Skip cameras that are running and delivering live frames.
                if cam.is_healthy():
                    continue
                # Skip cameras that are disabled by config (-1) and were never
                # intended to be used — unless auto-detect is on and a second
                # device is now available.
                if cam.camera_id < 0:
                    # Check if a new device appeared that we could use.
                    from src.hardware.camera import list_capture_devices, IS_LINUX
                    if not IS_LINUX:
                        continue
                    available = list_capture_devices(probe_read=False)
                    # Quick check: is there a device we haven't claimed?
                    left_id = self.camera_manager.left_camera.camera_id
                    used = {left_id} if left_id >= 0 else set()
                    spare = next((i for i in available if i not in used), -1)
                    if spare < 0:
                        continue
                    # Re-enable this camera with the spare device.
                    print(f"Camera{cam_num}: new device /dev/video{spare} detected, "
                          f"auto-enabling {side} feed")
                    cam.camera_id = spare
                    cam.display_name = f"Camera{cam_num}"

                # Try to re-open the camera.
                if cam.start():
                    print(f"Camera{cam_num}: reconnected successfully")
                    self.log_message(f"Camera{cam_num} reconnected")
                    # Restart preview if we're not scanning (during a scan the
                    # detection thread will pick up the new frames automatically).
                    if not self.is_scanning and self.preview_thread is None:
                        self.start_preview()
                else:
                    # Camera still not available — keep the placeholder.
                    if label.text() != f"Camera{cam_num} not detected":
                        label.setPixmap(QPixmap())
                        label.setText(f"Camera{cam_num} not detected")
        except Exception as e:
            # Never let the hot-plug monitor crash the app.
            pass

    def update_detection(self, count: int, confidence: float, activity: str,
                         left_count: int = 0, right_count: int = 0):
        # Ignore detection updates outside of an active scan. The DetectionThread
        # is stopped when the scan ends, but queued signals may still fire after
        # is_scanning is set to False — discard them so the UI does not show
        # phantom counts or log detections after the scan has stopped.
        if not self.is_scanning:
            return

        # Only show count/confidence during an active scan with cameras running.
        if self.count_label:
            self.count_label.setText(f"Weevil Count: {count}")
            self.confidence_label.setText(
                f"Avg Confidence: {confidence * 100:.1f}%" if confidence else "Avg Confidence: --")

        # Track scan metadata
        if count > self.scan_max_count:
            self.scan_max_count = count

        current_time = datetime.now()
        # "Significant" means a noticeably high reading, not merely a non-zero one.
        is_significant = count >= self.high_weevil_threshold
        
        # Capture annotated snapshots from BOTH camera feeds when the model
        # detects Sitophilus Oryzae (count > 0), rate-limited by
        # IMAGE_CAPTURE_COOLDOWN (default 10s). This ensures every detection
        # event with bounding boxes is captured and stored — in the scan
        # folder, the database, and the emailed zip — rather than capturing
        # empty frames on a fixed timer.
        #
        # Raw frames (no boxes) are also captured periodically even when no
        # weevils are detected, so the retraining dataset includes negative
        # samples (frames where the model should have detected but missed,
        # or true negatives). These raw-only captures use a longer cooldown
        # (3x the normal cooldown) to avoid flooding the archive with empty
        # frames.
        capture_cooldown_elapsed = (self.last_image_capture_time is None or
                                     (current_time - self.last_image_capture_time).total_seconds() >= self.image_capture_cooldown)
        should_capture = self.is_scanning and count > 0 and capture_cooldown_elapsed
        # Raw-only capture (no detections): longer cooldown, only raw frames.
        raw_only_cooldown = self.image_capture_cooldown * 3
        last_raw_capture = getattr(self, '_last_raw_only_capture_time', None)
        raw_only_elapsed = (last_raw_capture is None or
                            (current_time - last_raw_capture).total_seconds() >= raw_only_cooldown)
        should_capture_raw_only = (self.is_scanning and count == 0 and raw_only_elapsed
                                    and capture_cooldown_elapsed)
        
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
        log_entry = self.logger.log_detection(count, self.is_after_mixing, activity)

        # Recommendation is only shown after the scan completes (in
        # on_scan_duration_elapsed / stop_scan), not during the scan.

        if not (should_log or should_capture or should_capture_raw_only):
            return

        # Persist the detection row, then attach any captured images to it
        detection_id = None
        if self.current_scan_id:
            detection_id = self.db.add_detection(
                self.current_scan_id,
                log_entry.timestamp,
                count,
                log_entry.recommendation,
                activity,
                left_count=left_count,
                right_count=right_count,
            )
            self.scan_detection_count += 1

        if should_capture:
            self.capture_detection_images(count, confidence, detection_id,
                                          left_count=left_count, right_count=right_count)
        elif should_capture_raw_only:
            self._last_raw_only_capture_time = current_time
            self.capture_detection_images(0, 0.0, detection_id,
                                          left_count=0, right_count=0,
                                          raw_only=True)

        if should_capture:
            self.last_image_capture_time = current_time
        
        if should_log:
            marker = "HIGH COUNT " if is_significant else ""
            self.log_message(f"{marker}{log_entry.timestamp} - Count: {count}, "
                             f"Rec: {log_entry.recommendation}")
            self.last_log_time = current_time
            if should_log_detection:
                self.last_detection_log_time = current_time
    
    def capture_detection_images(self, count: int, confidence: float = 0.0,
                                detection_id: Optional[int] = None,
                                left_count: int = 0, right_count: int = 0,
                                raw_only: bool = False) -> list:
        """Store annotated snapshots of the detected rice weevils in the database (and on disk).

        The JPEG bytes go into scan_images so the emailing system can rebuild the report
        archive from the database alone. Bounding boxes are drawn directly on the
        current camera frame using the latest detection results — not from the
        preview thread's cached annotated frame, which may not have been updated
        yet due to thread timing. Each stored image records its own camera's
        weevil count (left_count/right_count) so the per-feed reading is
        preserved alongside the deduplicated max(left, right) total.

        When raw_only=True, only raw frames (no boxes) from both cameras are
        stored — used for periodic captures during quiet periods so the
        retraining dataset includes negative samples.
        """
        if not self.current_scan_id:
            return []

        timestamp = datetime.now()
        stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        prefix = "high_weevil" if count >= self.high_weevil_threshold else "weevil"
        quality = int(os.getenv('IMAGE_JPEG_QUALITY', '85'))
        saved = []

        # Get the current raw frames and detection results so we can draw
        # bounding boxes directly. This avoids relying on latest_annotated_*
        # which is set by the preview thread and may be stale or None. Each
        # tuple carries its own camera's weevil count so the per-feed reading
        # is stored on the image row alongside the deduplicated total.
        #
        # For each camera we also build a "preprocessed" variant: the CLAHE-
        # equalized frame that is the model's actual input, with the same
        # bounding boxes drawn on it. Storing it gives an audit trail of what
        # the detector saw (useful when reviewing false positives/negatives),
        # and because it goes into scan_images it is included in the emailed
        # zip automatically. Preprocessing is captured BEFORE the raw frame is
        # mutated by draw_detections so the variant reflects the true input.
        #
        # We also store a "raw" variant: the original camera frame WITHOUT any
        # bounding boxes drawn on it. This is essential for retraining — if
        # the detection missed weevils or placed boxes incorrectly, the raw
        # frame can be manually annotated and added to the training dataset.
        store_preprocessed = self.store_preprocessed_images and self.detector.use_clahe
        frames_with_detections = []

        # raw_only mode: capture raw frames from BOTH cameras regardless of
        # detection count. Used for periodic captures during quiet periods so
        # the retraining dataset includes negative samples (frames where the
        # model should find nothing, or missed weevils that can be manually
        # annotated later).
        if raw_only:
            left_frame = self._latest_detected_frame_left
            if left_frame is None:
                left_frame = self.camera_manager.get_left_frame()
            right_frame = self._latest_detected_frame_right
            if right_frame is None:
                right_frame = self.camera_manager.get_right_frame()
            if left_frame is not None:
                frames_with_detections.append(("left_raw", left_frame.copy(), 0))
            if right_frame is not None:
                frames_with_detections.append(("right_raw", right_frame.copy(), 0))
            prefix = "baseline"
            # Fall through to the shared storage loop below.
        else:
            # Normal mode: capture raw + annotated + preprocessed for cameras with detections.

            # Use the EXACT frames that were passed to detector.detect() — not fresh
            # frames from the camera manager. With 2x2 tiling each detection cycle
            # takes ~2-4 seconds, during which the weevil moves. If we grabbed a new
            # frame here, the bounding boxes (from the old frame) would not match
            # the weevil's current position. Using the detected frame ensures the
            # stored annotated image has boxes correctly aligned on the weevils.
            left_frame = self._latest_detected_frame_left
            if left_frame is not None and left_count > 0:
                # Store the raw frame (no boxes) for manual retraining annotation.
                frames_with_detections.append(("left_raw", left_frame.copy(), left_count))
            left_pre = self.detector.preprocess_frame(left_frame) if store_preprocessed else None
            if left_pre is left_frame:
                left_pre = left_frame.copy()
            if self._latest_detection_left is not None:
                annotated = self.detector.draw_detections(left_frame, self._latest_detection_left)
            else:
                annotated = left_frame
            frames_with_detections.append(("left", annotated, left_count))
            if left_pre is not None:
                if self._latest_detection_left is not None:
                    left_pre = self.detector.draw_detections(left_pre, self._latest_detection_left)
                frames_with_detections.append(("left_preprocessed", left_pre, left_count))

            right_frame = self._latest_detected_frame_right
            if right_frame is not None and right_count > 0:
                # Store the raw frame (no boxes) for manual retraining annotation.
                frames_with_detections.append(("right_raw", right_frame.copy(), right_count))
                right_pre = self.detector.preprocess_frame(right_frame) if store_preprocessed else None
                if right_pre is right_frame:
                    right_pre = right_frame.copy()
                if self._latest_detection_right is not None:
                    annotated = self.detector.draw_detections(right_frame, self._latest_detection_right)
                else:
                    annotated = right_frame
                frames_with_detections.append(("right", annotated, right_count))
                if right_pre is not None:
                    if self._latest_detection_right is not None:
                        right_pre = self.detector.draw_detections(right_pre, self._latest_detection_right)
                    frames_with_detections.append(("right_preprocessed", right_pre, right_count))

        for side, frame, side_count in frames_with_detections:
            filename = f"{prefix}_{side}_{stamp}_count{side_count}.jpg"
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
                    weevil_count=side_count,
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
            elif count > 0:
                self.log_message(f"Weevil count {count} - "
                                 f"stored {len(saved)} image(s) in database")
            else:
                self.log_message(f"Baseline raw capture - "
                                 f"stored {len(saved)} image(s) in database")
        return saved

    def update_clock(self):
        now = datetime.now()
        # Format: June 25, 2026 9:31:45 PM
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        # The header clock stays visible on every tab, so a separate date/time
        # panel on Live Feed would only duplicate it.
        self.header_clock_label.setText(f"{date_str}   {time_str}")

    def log_message(self, message: str):
        # Buffer the line; _flush_log_buffer appends it to the QTextEdit in a
        # batched update so a burst of messages does not stall the GUI.
        self._log_buffer.append(message)

    def _on_console_line(self, line: str):
        """Slot for ConsoleLogStream.line_printed — mirrors terminal output."""
        self._log_buffer.append(line)

    def _flush_log_buffer(self):
        """Drain buffered log lines into the System Log in one batch.

        Coalescing many rapid prints (e.g. a failing camera) into a single
        QTextEdit update keeps the GUI responsive. The document is also capped
        so append() stays cheap instead of growing without bound and slowing
        every subsequent append until the UI hangs.
        """
        if not self._log_buffer:
            return
        # One append() for the whole burst -> one document reflow instead of
        # one per line.
        self.log_text.append("\n".join(self._log_buffer))
        self._log_buffer.clear()
        self._trim_log_text()
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _trim_log_text(self):
        """Keep the System Log to at most MAX_LOG_LINES blocks.

        QTextEdit.append() is O(n) in the document size, so an unbounded log
        makes every append slower until the UI hangs. Dropping the oldest
        blocks from the top keeps the document small and append() cheap.
        """
        doc = self.log_text.document()
        max_lines = self.MAX_LOG_LINES
        excess = doc.blockCount() - max_lines
        if excess <= 0:
            return
        cursor = QTextCursor(doc)
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.Start)
        # Extend the selection to the end of the (excess)th block so the
        # removed range spans whole blocks only.
        for _ in range(excess):
            cursor.movePosition(QTextCursor.NextBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.endEditBlock()

    def closeEvent(self, event):
        if self.is_scanning:
            self.stop_scan()
        # Stop the live preview loop and release the cameras on exit
        self.stop_preview()
        self.camera_manager.stop()
        # Let any in-flight report email finish so the scan report is not lost on exit.
        # SMTP can take up to 60s (the timeout in send_email), so wait long enough.
        pending = list(self.email_threads)
        if pending:
            print(f"Waiting for {len(pending)} email(s) to finish sending before exit...")
            for thread in pending:
                thread.wait(120000)  # 2 minutes max per email
            print("All emails completed.")
        # Stop the camera hot-plug monitor.
        self.camera_monitor_timer.stop()
        # Stop the log flush timer and drain any buffered lines so the last
        # messages are written before stdout/stderr are restored below.
        self._log_flush_timer.stop()
        self._flush_log_buffer()
        # Restore original stdout/stderr so post-shutdown messages go to the real terminal
        sys.stdout = self._console_stream._original
        sys.stderr = self._console_stream_err._original
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Show loading screen first. The MainWindow is created AFTER the loading
    # screen finishes its 5-second display, so the user sees a smooth splash
    # before the main app appears. The loading screen duration is calibrated to
    # match the time MainWindow takes to initialize (model load, camera probe).
    logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'logo.png')
    from src.gui.loading_screen import LoadingScreen
    loading_screen = LoadingScreen(logo_path)
    loading_screen.show()

    # Create and show the main window only after the loading screen completes.
    # The loading screen's 5-second timer runs in the Qt event loop, so the UI
    # stays responsive. When it fires, the MainWindow is created (which takes
    # ~3-5s for model load + camera probe), then shown.
    loading_screen.loading_complete.connect(lambda: show_main_window(loading_screen))

    sys.exit(app.exec_())


def show_main_window(loading_screen):
    """Transition from loading screen to main window.

    The MainWindow is created here so hardware initialization only happens once,
    after the loading screen closes. The loading screen closes first so the user
    doesn't see two windows at once."""
    loading_screen.close()
    loading_screen.deleteLater()
    window = MainWindow()
    window.show()

    # Install a global excepthook so unhandled exceptions in QThreads print a
    # full traceback instead of silently killing the process with just
    # "Unhandled Python exception". This makes crashes debuggable.
    def _global_excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        import traceback
        print(f"Unhandled exception: {exc_value}")
        traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.excepthook = _global_excepthook


if __name__ == '__main__':
    main()
