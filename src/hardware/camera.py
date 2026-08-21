import cv2
import threading
import time
from typing import Optional, Callable
import numpy as np


class Camera:
    """A single USB/CSI camera with live-signal validation.

    Opening a device is not proof that a real camera is attached.  On Windows a
    virtual camera (NVIDIA Broadcast, OBS, driver placeholders) will open and
    happily hand back frames that are either completely blank or the same frozen
    image over and over.  Running YOLO on those produces phantom detections, so
    every frame is checked for signs of life before it is published.
    """

    # A live scene always has some pixel spread.  A solid colour / black frame
    # has essentially none, so anything below this std-dev is treated as blank.
    BLANK_STD_THRESHOLD = 4.0
    # A real sensor always has at least a little noise between frames.  This many
    # byte-identical frames in a row means the source is frozen, not live.
    FROZEN_FRAME_LIMIT = 15

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_callback: Optional[Callable] = None
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.current_frame = None
        self.last_frame_time: Optional[float] = None
        # Live-signal state, all guarded by self.lock
        self.signal_live = False
        self.signal_reason = "not started"
        self._last_signature: Optional[np.ndarray] = None
        self._identical_frames = 0

    def initialize(self) -> bool:
        # A negative id means "this camera is intentionally not installed", which
        # lets a single-camera rig run without the code pretending to open device 1.
        if self.camera_id < 0:
            print(f"Camera {self.camera_id} disabled by configuration")
            return False
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

    @staticmethod
    def _signature(frame: np.ndarray) -> np.ndarray:
        """Cheap fingerprint of a frame, used to spot a frozen source."""
        return frame[::16, ::16].copy()

    def start(self, frame_callback: Optional[Callable] = None) -> bool:
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
                if not ret or frame is None or frame.size == 0:
                    # Clear the stale frame so the detection thread does not
                    # run inference on a frame from a camera that lost its signal.
                    with self.lock:
                        self.current_frame = None
                        self.signal_live = False
                        self.signal_reason = "no frame from device"
                    print(f"Camera {self.camera_id}: failed to read frame")
                    time.sleep(0.1)
                    continue

                signature = self._signature(frame)
                is_blank = float(signature.std()) < self.BLANK_STD_THRESHOLD

                with self.lock:
                    if self._last_signature is not None and np.array_equal(signature, self._last_signature):
                        self._identical_frames += 1
                    else:
                        self._identical_frames = 0
                    self._last_signature = signature
                    is_frozen = self._identical_frames >= self.FROZEN_FRAME_LIMIT

                    was_live = self.signal_live
                    if is_blank:
                        self.signal_live = False
                        self.signal_reason = "blank frame (no image data)"
                    elif is_frozen:
                        self.signal_live = False
                        self.signal_reason = "frozen frame (virtual or disconnected camera)"
                    else:
                        self.signal_live = True
                        self.signal_reason = "live"

                    self.current_frame = frame.copy()
                    self.last_frame_time = time.time()
                    became_dead = was_live and not self.signal_live
                    became_live = not was_live and self.signal_live
                    reason = self.signal_reason

                # Log transitions only, so a dead camera does not spam the log
                if became_dead:
                    print(f"Camera {self.camera_id}: signal lost - {reason}")
                elif became_live:
                    print(f"Camera {self.camera_id}: live signal detected")

                if self.frame_callback:
                    self.frame_callback(frame)
            except Exception as e:
                print(f"Capture loop error: {e}")
                with self.lock:
                    self.signal_live = False
                    self.signal_reason = f"capture error: {e}"
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
        with self.lock:
            self.current_frame = None
            self.signal_live = False
            self.signal_reason = "stopped"
            self._last_signature = None
            self._identical_frames = 0

    def is_running(self) -> bool:
        return self.running

    def is_healthy(self, max_stale_seconds: float = 3.0) -> bool:
        """True only when the camera is delivering a genuinely live picture.

        Three things must hold: the capture thread is running, a frame arrived
        within the last *max_stale_seconds*, and that frame passed the blank /
        frozen checks in the capture loop.  The detection thread uses this to
        decide whether a camera is worth running inference on, which is what
        stops a virtual or unplugged camera from producing phantom detections.
        """
        if not self.running:
            return False
        with self.lock:
            if self.current_frame is None or self.last_frame_time is None:
                return False
            if not self.signal_live:
                return False
            return (time.time() - self.last_frame_time) <= max_stale_seconds

    def get_signal_reason(self) -> str:
        with self.lock:
            return self.signal_reason

    def __del__(self):
        self.stop()


class DualCameraManager:
    """Manages two cameras (left and right) for stereo vision"""
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1,
                 width: int = 640, height: int = 480, fps: int = 30):
        # Guard against both IDs pointing at the same physical device. On Windows,
        # OpenCV may expose the same camera under multiple indices (e.g. MSMF and
        # DShow backends), which would make both feeds show identical pictures.
        # If the IDs match and the camera is not disabled (-1), disable the right
        # camera so only the left feed is shown.
        if left_camera_id == right_camera_id and left_camera_id >= 0:
            print(f"WARNING: Left and right camera IDs are both {left_camera_id}. "
                  f"Disabling right camera to avoid duplicate feeds. "
                  f"Set RIGHT_CAMERA_ID to a different device in config.env.")
            right_camera_id = -1
        self.left_camera = Camera(left_camera_id, width, height, fps)
        self.right_camera = Camera(right_camera_id, width, height, fps)
        self.running = False

    def start(self, left_callback: Optional[Callable] = None, right_callback: Optional[Callable] = None) -> bool:
        left_started = self.left_camera.start(left_callback)
        right_started = self.right_camera.start(right_callback)
        # The scan can proceed as long as at least one camera is live.  The
        # detection thread checks is_healthy() per camera so a missing feed does
        # not produce phantom detections.
        self.running = left_started or right_started
        return self.running

    def stop(self):
        self.left_camera.stop()
        self.right_camera.stop()
        self.running = False

    def get_left_frame(self) -> Optional[np.ndarray]:
        return self.left_camera.get_frame()

    def get_right_frame(self) -> Optional[np.ndarray]:
        return self.right_camera.get_frame()

    def is_running(self) -> bool:
        return self.running and (self.left_camera.is_running() or self.right_camera.is_running())

    def is_left_healthy(self) -> bool:
        return self.left_camera.is_healthy()

    def is_right_healthy(self) -> bool:
        return self.right_camera.is_healthy()

    def wait_for_signal(self, timeout: float = 3.0) -> tuple:
        """Give the cameras a moment to prove they have a live picture.

        The frozen-frame check needs several frames before it can tell a static
        virtual camera from a real one, so callers that want an immediate verdict
        (such as the GUI when a scan starts) should wait here first.

        Returns (left_healthy, right_healthy).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_left_healthy() and self.is_right_healthy():
                break
            time.sleep(0.1)
        return self.is_left_healthy(), self.is_right_healthy()

    def __del__(self):
        self.stop()
