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
