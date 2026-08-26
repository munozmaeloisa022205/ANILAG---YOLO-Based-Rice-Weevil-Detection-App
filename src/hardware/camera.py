import cv2
import glob
import os
import sys
import threading
import time
from typing import Optional, Callable, List
import numpy as np

IS_LINUX = sys.platform.startswith('linux')


def _default_fourcc() -> str:
    """Pixel format to request from the driver.

    On the Raspberry Pi 5 both USB webcams share a single USB controller. Two
    uncompressed YUYV streams at 640x480x30 need ~2 x 110 Mbit/s of isochronous
    bandwidth, which the controller refuses to allocate - the second camera then
    fails with 'VIDIOC_STREAMON: No space left on device' or silently drops to a
    few FPS. MJPG is compressed in the camera, so both streams fit comfortably.

    Windows webcams are not on a shared bandwidth budget here and their drivers
    are pickier about formats, so they keep the driver default instead.

    An unset OR EMPTY CAMERA_FOURCC means "use the platform default" - a blank
    value in config.env must not silently cost the Pi its MJPG. Use
    CAMERA_FOURCC=NONE to explicitly force the driver default everywhere.
    """
    configured = os.getenv('CAMERA_FOURCC', '').strip().upper()
    if configured in ('NONE', 'DEFAULT'):
        return ''
    if configured:
        return configured
    return 'MJPG' if IS_LINUX else ''


def list_capture_devices(probe_read: bool = True) -> List[int]:
    """Indices of /dev/video* nodes that can actually deliver frames (Linux only).

    A UVC webcam on Raspberry Pi OS registers TWO nodes: an even one that streams
    video and an odd one that only carries UVC metadata. So a rig with two cameras
    exposes /dev/video0..3, where 0 and 2 are the cameras and 1 and 3 are metadata.
    Configuring LEFT_CAMERA_ID=0 / RIGHT_CAMERA_ID=1 therefore points the "right"
    camera at a metadata node that never produces an image.

    There is no sysfs attribute exposing V4L2 capabilities, so each node is opened
    and asked for one frame - the only definitive test.
    """
    if not IS_LINUX:
        return []

    indices = []
    for path in sorted(glob.glob('/dev/video*')):
        suffix = path[len('/dev/video'):]
        if not suffix.isdigit():
            continue
        index = int(suffix)
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            if not probe_read:
                indices.append(index)
                continue
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                indices.append(index)
        except Exception:
            continue
        finally:
            cap.release()
    return indices


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

    # A freshly opened UVC device needs a moment before the first frame arrives,
    # so the open-time liveness probe retries instead of failing immediately.
    OPEN_PROBE_ATTEMPTS = 10
    OPEN_PROBE_DELAY = 0.2

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = _default_fourcc()
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
            # Ask for V4L2 explicitly on Linux. With CAP_ANY, OpenCV may pick its
            # GStreamer backend, which ignores the CAP_PROP_* settings below and
            # adds latency on the Pi.
            # On Windows use CAP_DSHOW instead of CAP_ANY: CAP_ANY may pick
            # different backends for different indices (MSMF for one, DSHOW for
            # another), causing inconsistent frame quality. CAP_DSHOW also
            # avoids NVIDIA Broadcast's MSMF hook which re-publishes a real
            # camera as a virtual one, creating duplicate feeds.
            backend = cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_DSHOW
            self.cap = cv2.VideoCapture(self.camera_id, backend)
            if not self.cap.isOpened():
                print(f"Camera {self.camera_id}: device could not be opened")
                self.cap = None
                return False

            # FOURCC must be set BEFORE the resolution: the driver validates the
            # requested frame size against the currently selected pixel format.
            if self.fourcc and len(self.fourcc) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Keep the driver queue at one frame. Otherwise the detection thread,
            # which is slower than the camera, reads progressively staler frames.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._probe_first_frame():
                # On Linux this is the usual symptom of pointing at a UVC metadata
                # node rather than a capture node.
                print(f"Camera {self.camera_id}: opened but delivered no frame")
                self.cap.release()
                self.cap = None
                return False

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera {self.camera_id}: {actual_w}x{actual_h} "
                  f"{self.fourcc or 'default'} @ {self.cap.get(cv2.CAP_PROP_FPS):g} FPS")
            return True
        except Exception as e:
            print(f"Camera initialization error: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

    def _probe_first_frame(self) -> bool:
        """Confirm the device really streams images before declaring it usable."""
        for _ in range(self.OPEN_PROBE_ATTEMPTS):
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
            time.sleep(self.OPEN_PROBE_DELAY)
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

                    self.current_frame = frame
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

    def get_frame_raw(self) -> Optional[np.ndarray]:
        """Return the current frame without copying.

        Faster than get_frame() — the .copy() in get_frame() is a full-frame
        memcpy (640x480x3 = ~900 KB) that adds up at 30 FPS. Safe as long as
        the caller only reads the array (never mutates it). The capture loop
        replaces current_frame atomically under self.lock, so the returned
        array's backing memory stays valid until the next cap.read() overwrites
        the buffer — but since OpenCV's read() returns a new array each time,
        the previous array is not modified in place.
        """
        with self.lock:
            return self.current_frame

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

    # Two different indices can still be one physical camera. Virtual-camera
    # software (NVIDIA Broadcast, OBS) re-publishes a real webcam under an index
    # of its own, and Windows sometimes exposes the same device through both the
    # MSMF and DShow backends. The panes then show the same picture twice, which
    # silently DOUBLE-COUNTS every weevil in the combined total, so the two feeds
    # are compared once at startup.
    #
    # Independent sensors never agree this closely: even aimed at the same scene
    # they differ by their own read noise, so a mean absolute difference this
    # small means one source is being copied.
    DUPLICATE_DIFF_THRESHOLD = 1.0
    DUPLICATE_SAMPLES = 5
    DUPLICATE_SAMPLE_DELAY = 0.15

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

        left_camera_id, right_camera_id = self._resolve_ids(left_camera_id, right_camera_id)
        self.left_camera = Camera(left_camera_id, width, height, fps)
        self.right_camera = Camera(right_camera_id, width, height, fps)
        self.running = False

    @staticmethod
    def _resolve_ids(left_id: int, right_id: int) -> tuple:
        """Map the configured indices onto nodes that actually stream (Linux only).

        Raspberry Pi OS gives each UVC webcam a capture node AND a metadata node,
        so the obvious 0/1 pairing usually lands the right camera on camera 0's
        metadata node. Probing once here means config.env does not have to encode
        the kernel's enumeration order, which changes with USB port and boot.

        Set CAMERA_AUTO_DETECT=false to use the configured indices verbatim.
        """
        if not IS_LINUX:
            return left_id, right_id
        if os.getenv('CAMERA_AUTO_DETECT', 'true').lower() not in ('true', '1', 'yes', 'on'):
            return left_id, right_id

        available = list_capture_devices()
        if not available:
            print("WARNING: no streaming /dev/video* capture device found. Check "
                  "'ls /dev/video*' and that the user is in the 'video' group.")
            return left_id, right_id
        print(f"Capture-capable video devices: {available}")

        resolved = []
        for name, configured in (('left', left_id), ('right', right_id)):
            if configured < 0:
                resolved.append(-1)
                continue
            if configured in available and configured not in resolved:
                resolved.append(configured)
                continue
            fallback = next((i for i in available if i not in resolved), -1)
            if fallback < 0:
                print(f"WARNING: no capture device left for the {name} camera. "
                      f"Only {len(available)} camera(s) detected; disabling it.")
            else:
                print(f"Remapping {name} camera {configured} -> /dev/video{fallback} "
                      f"(device {configured} does not stream video)")
            resolved.append(fallback)
        return resolved[0], resolved[1]

    def _wait_for_both_frames(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.get_left_frame() is not None and self.get_right_frame() is not None:
                return True
            time.sleep(0.1)
        return False

    def _sources_are_duplicates(self) -> bool:
        """True when both cameras are demonstrably publishing the same picture."""
        if not self._wait_for_both_frames():
            return False

        for _ in range(self.DUPLICATE_SAMPLES):
            left = self.get_left_frame()
            right = self.get_right_frame()
            if left is None or right is None or left.shape != right.shape:
                return False
            # Two blank frames match trivially, so they prove nothing about the
            # source - a covered or unlit camera would otherwise look like a
            # duplicate. The liveness check already reports these as no-signal.
            if float(left.std()) < Camera.BLANK_STD_THRESHOLD:
                return False
            difference = np.abs(left.astype(np.int16) - right.astype(np.int16)).mean()
            if difference > self.DUPLICATE_DIFF_THRESHOLD:
                return False
            time.sleep(self.DUPLICATE_SAMPLE_DELAY)
        return True

    def start(self, left_callback: Optional[Callable] = None, right_callback: Optional[Callable] = None) -> bool:
        left_started = self.left_camera.start(left_callback)
        right_started = self.right_camera.start(right_callback)
        # The scan can proceed as long as at least one camera is live.  The
        # detection thread checks is_healthy() per camera so a missing feed does
        # not produce phantom detections.
        self.running = left_started or right_started

        check_enabled = os.getenv('CAMERA_DUPLICATE_CHECK', 'true').lower() in (
            'true', '1', 'yes', 'on')
        if left_started and right_started and check_enabled and self._sources_are_duplicates():
            print(f"WARNING: cameras {self.left_camera.camera_id} and "
                  f"{self.right_camera.camera_id} are delivering an identical picture, so "
                  f"they are the same physical camera. Disabling the right feed to avoid "
                  f"double-counting. One of the two indices is most likely a virtual camera "
                  f"(NVIDIA Broadcast, OBS); set RIGHT_CAMERA_ID in config.env to the other "
                  f"real device.")
            self.right_camera.stop()

        return self.running

    def stop(self):
        self.left_camera.stop()
        self.right_camera.stop()
        self.running = False

    def get_left_frame(self) -> Optional[np.ndarray]:
        return self.left_camera.get_frame()

    def get_right_frame(self) -> Optional[np.ndarray]:
        return self.right_camera.get_frame()

    def get_left_frame_raw(self) -> Optional[np.ndarray]:
        return self.left_camera.get_frame_raw()

    def get_right_frame_raw(self) -> Optional[np.ndarray]:
        return self.right_camera.get_frame_raw()

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
