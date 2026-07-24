import cv2
import threading
from typing import Optional, Callable
import numpy as np


class Camera:
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


class DualCameraManager:
    """Manages two cameras (left and right) for stereo vision"""
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1,
                 width: int = 640, height: int = 480, fps: int = 30):
        self.left_camera = Camera(left_camera_id, width, height, fps)
        self.right_camera = Camera(right_camera_id, width, height, fps)
        self.running = False

    def start(self, left_callback: Optional[Callable] = None, right_callback: Optional[Callable] = None) -> bool:
        left_started = self.left_camera.start(left_callback)
        right_started = self.right_camera.start(right_callback)
        self.running = left_started and right_started
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
        return self.running and self.left_camera.is_running() and self.right_camera.is_running()

    def __del__(self):
        self.stop()
