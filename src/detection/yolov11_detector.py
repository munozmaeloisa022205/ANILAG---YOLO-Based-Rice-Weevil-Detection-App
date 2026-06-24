import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import threading


class DetectionResult:
    def __init__(self, boxes: List, confidences: List[float], class_ids: List[int], class_names: List[str]):
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids
        self.class_names = class_names
        self.count = len(boxes)

    def has_detections(self) -> bool:
        return self.count > 0


class YOLOv11Detector:
    def __init__(self, model_path: str = 'yolov11n.pt', confidence_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model: Optional[YOLO] = None
        self.class_names = []
        self.lock = threading.Lock()
        self.initialized = False

    def initialize(self) -> bool:
        try:
            with self.lock:
                self.model = YOLO(self.model_path)
                self.class_names = self.model.names
                self.initialized = True
                print(f"YOLOv11 model loaded from {self.model_path}")
                print(f"Classes: {self.class_names}")
                return True
        except Exception as e:
            print(f"YOLO model initialization error: {e}")
            return False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.initialized or self.model is None:
            return DetectionResult([], [], [], [])

        try:
            with self.lock:
                results = self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            
            boxes = []
            confidences = []
            class_ids = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                        confidences.append(float(box.conf[0].cpu().numpy()))
                        class_ids.append(int(box.cls[0].cpu().numpy()))
            
            return DetectionResult(boxes, confidences, class_ids, self.class_names)
        except Exception as e:
            print(f"Detection error: {e}")
            return DetectionResult([], [], [], [])

    def draw_detections(self, frame: np.ndarray, detection: DetectionResult, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for i, (box, conf, cls_id) in enumerate(zip(detection.boxes, detection.confidences, detection.class_ids)):
            x1, y1, x2, y2 = box
            class_name = detection.class_names[cls_id] if cls_id < len(detection.class_names) else f"Class {cls_id}"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw count
        count_text = f"Count: {detection.count}"
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

    def get_rice_weevil_count(self, detection: DetectionResult, rice_weevil_class_id: Optional[int] = None) -> int:
        if rice_weevil_class_id is not None:
            return sum(1 for cls_id in detection.class_ids if cls_id == rice_weevil_class_id)
        return detection.count
