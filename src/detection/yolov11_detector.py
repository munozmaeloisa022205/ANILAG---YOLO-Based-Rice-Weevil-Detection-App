import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import threading
import os
import platform

# Class names that indicate the fine-tuned rice weevil model. Anything else means a
# generic COCO checkpoint (e.g. the stock yolo11n.pt) was loaded by mistake.
WEEVIL_NAME_HINTS = ('sitophilus', 'weevil', 'oryzae', 'bukbok')

# A few COCO class names used to recognise an un-finetuned checkpoint.
COCO_MARKERS = ('person', 'bicycle', 'car', 'traffic light')

# Training settings of models/sitophilus_oryzae_v2-3_best.pt, taken from
# AI-MODEL/runs/detect/sitophilus_oryzae_v2-3/args.yaml (verified: the shipped
# checkpoint is byte-identical to that run's weights/best.pt).
TRAINED_IMGSZ = 640
TRAINED_IOU = 0.7
TRAINED_CLASS_NAMES = ('Sitophilus Oryzae',)
# Final validation metrics from that run's results.csv (epoch 157).
TRAINED_METRICS = {
    'run': 'sitophilus_oryzae_v2-3',
    'epochs': 157,
    'mAP50': 0.958,
    'mAP50-95': 0.717,
    'precision': 0.944,
    'recall': 0.954,
}


class DetectionResult:
    def __init__(self, boxes: List, confidences: List[float], class_ids: List[int], class_names: List[str],
                 inference_ms: float = 0.0):
        self.boxes = boxes
        self.confidences = confidences
        self.class_ids = class_ids
        self.class_names = class_names
        self.count = len(boxes)
        self.inference_ms = inference_ms

    def has_detections(self) -> bool:
        return self.count > 0

    def average_confidence(self) -> Optional[float]:
        return sum(self.confidences) / len(self.confidences) if self.confidences else None


def _is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except Exception:
        return platform.machine() in ('aarch64', 'armv7l')


class YOLOv11Detector:
    def __init__(self, model_path: str = 'models/sitophilus_oryzae_v2-3_best.pt', confidence_threshold: float = 0.5, iou_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model: Optional[YOLO] = None
        self.class_names = []
        self.lock = threading.Lock()
        self.initialized = False
        # Raspberry Pi 5 optimizations
        self._is_pi = _is_raspberry_pi()
        self._device = 'cpu'  # Pi 5 has no CUDA GPU
        # Must match the imgsz the weights were trained at (640 for sitophilus_oryzae_v2-3)
        # or recall on small weevils drops. Lower only if the Pi 5 cannot keep up.
        self._imgsz = int(os.getenv('YOLO_IMGSZ', str(TRAINED_IMGSZ)))
        self._trained_imgsz = int(os.getenv('MODEL_TRAINED_IMGSZ', str(TRAINED_IMGSZ)))
        self._max_det = int(os.getenv('YOLO_MAX_DET', '300'))  # matches the training/val default
        self.use_ncnn = False
        self.loaded_path = model_path
        # The training dataset was preprocessed with "auto-contrast via adaptive
        # equalization" (see AI-MODEL/.../README.roboflow.txt). Applying the same
        # equalization to live frames closes that domain gap, which improves both
        # precision and recall. CLAHE on the LAB L-channel is the OpenCV equivalent.
        self.use_clahe = os.getenv('YOLO_CLAHE', 'true').lower() in ('true', '1', 'yes', 'on')
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Only these class ids are counted as rice weevils; None means count every class.
        self.weevil_class_ids: Optional[List[int]] = None
        self.is_generic_model = False
        self.warnings: List[str] = []
        # Rolling inference statistics for the UI
        self.last_inference_ms = 0.0
        self.avg_inference_ms = 0.0
        self._inference_samples = 0

    @property
    def backend(self) -> str:
        return 'NCNN' if self.use_ncnn else 'PyTorch'

    @property
    def warning(self) -> Optional[str]:
        return '\n'.join(self.warnings) if self.warnings else None

    def add_warning(self, message: str):
        if message not in self.warnings:
            self.warnings.append(message)
            print(f"WARNING: {message}")

    def _find_ncnn_model(self) -> Optional[str]:
        """Return the NCNN export directory for this model if it exists."""
        base = os.path.splitext(self.model_path)[0]
        ncnn_dir = base + '_ncnn_model'
        if os.path.isdir(ncnn_dir) and os.path.exists(os.path.join(ncnn_dir, 'model.ncnn.bin')):
            return ncnn_dir
        return None

    @staticmethod
    def _ncnn_export_imgsz(ncnn_dir: str) -> Optional[int]:
        """Read the input size baked into an NCNN export from its metadata.yaml."""
        metadata_path = os.path.join(ncnn_dir, 'metadata.yaml')
        if not os.path.exists(metadata_path):
            return None
        def ints(text: str) -> List[int]:
            return [int(t) for t in ''.join(c if c.isdigit() else ' ' for c in text).split()]

        try:
            with open(metadata_path, 'r') as f:
                lines = f.read().splitlines()
        except OSError:
            return None

        for index, line in enumerate(lines):
            if not line.strip().startswith('imgsz:'):
                continue
            # Inline form: "imgsz: 640" or "imgsz: [640, 640]"
            sizes = ints(line.split(':', 1)[1])
            if sizes:
                return max(sizes)
            # Block form:
            #   imgsz:
            #   - 640
            #   - 640
            for following in lines[index + 1:]:
                if not following.strip().startswith('-'):
                    break
                sizes.extend(ints(following))
            return max(sizes) if sizes else None
        return None

    def _resolve_class_names(self):
        """Work out which class ids are rice weevils and warn on a non-finetuned model."""
        names = self.model.names
        if isinstance(names, dict):
            self.class_names = [names[k] for k in sorted(names)]
            items = sorted(names.items())
        else:
            self.class_names = list(names)
            items = list(enumerate(self.class_names))

        weevil_ids = [i for i, name in items if any(h in str(name).lower() for h in WEEVIL_NAME_HINTS)]
        if weevil_ids:
            self.weevil_class_ids = weevil_ids
        elif any(str(name).lower() in COCO_MARKERS for _, name in items):
            # Stock COCO checkpoint - counting its detections as weevils would be nonsense.
            self.is_generic_model = True
            self.weevil_class_ids = []
            self.add_warning(f"{os.path.basename(self.loaded_path)} is a generic COCO model "
                             f"({len(self.class_names)} classes, no rice weevil class). Detections are "
                             f"suppressed - set MODEL_PATH to the fine-tuned rice weevil model.")
        else:
            self.weevil_class_ids = None

    def _check_training_alignment(self):
        """Warn when inference settings drift from what the weights were trained with."""
        if self._imgsz != self._trained_imgsz:
            direction = "below" if self._imgsz < self._trained_imgsz else "above"
            self.add_warning(
                f"Inference runs at {self._imgsz}px, {direction} the {self._trained_imgsz}px the model was "
                f"trained at. Expect lower recall on small weevils than the "
                f"{TRAINED_METRICS['mAP50']:.3f} mAP50 measured during training.")
        if not self.is_generic_model and tuple(self.class_names) != TRAINED_CLASS_NAMES:
            self.add_warning(f"Loaded classes {self.class_names} differ from the trained "
                             f"class list {list(TRAINED_CLASS_NAMES)}.")

    def initialize(self) -> bool:
        try:
            with self.lock:
                # Prefer the NCNN export: roughly 5x faster than PyTorch on the Pi 5 ARM CPU.
                ncnn_path = self._find_ncnn_model()
                if ncnn_path and os.getenv('USE_NCNN', 'true').lower() in ('true', '1', 'yes', 'on'):
                    # The input size is baked into an NCNN export. Using it at a different
                    # imgsz silently degrades accuracy, so fall back to PyTorch instead.
                    export_imgsz = self._ncnn_export_imgsz(ncnn_path)
                    if export_imgsz is not None and export_imgsz != self._imgsz:
                        self.add_warning(
                            f"NCNN export was built at {export_imgsz}px but inference is configured for "
                            f"{self._imgsz}px, so the slower PyTorch model is being used. Re-export with: "
                            f"YOLO('{self.model_path}').export(format='ncnn', imgsz={self._imgsz})")
                    else:
                        try:
                            self.model = YOLO(ncnn_path, task='detect')
                            self.use_ncnn = True
                            self.loaded_path = ncnn_path
                        except Exception as e:
                            print(f"NCNN load failed, falling back to PyTorch: {e}")
                            self.model = None

                if self.model is None:
                    self.model = YOLO(self.model_path)
                    self.model.to(self._device)
                    self.use_ncnn = False
                    self.loaded_path = self.model_path

                self._resolve_class_names()
                self._check_training_alignment()
                self.initialized = True

                print(f"YOLOv11 model loaded ({self.backend}) from {self.loaded_path}")
                print(f"Running on: {'Raspberry Pi 5 (CPU)' if self._is_pi else 'CPU'}")
                print(f"Inference image size: {self._imgsz} (trained at {self._trained_imgsz})")
                print(f"Classes: {self.class_names}")
                return True
        except Exception as e:
            print(f"YOLO model initialization error: {e}")
            return False

    def export_to_ncnn(self) -> Optional[str]:
        """Export model to NCNN format for faster inference on Raspberry Pi 5 ARM CPU.
        Call this once on first setup. Returns path to exported model or None on failure."""
        try:
            model = YOLO(self.model_path)
            ncnn_path = model.export(format='ncnn', imgsz=self._imgsz)
            print(f"NCNN model exported to: {ncnn_path}")
            return ncnn_path
        except Exception as e:
            print(f"NCNN export error: {e}")
            return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Match the training preprocessing: adaptive-equalize contrast.

        The Roboflow export applied adaptive equalization to every training image,
        so raw camera frames are out-of-domain for the model. CLAHE is applied to
        the lightness channel only, preserving colour. Geometry is unchanged, so
        detection boxes map directly back onto the original frame.
        """
        if not self.use_clahe:
            return frame
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        except cv2.error:
            return frame

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.initialized or self.model is None or frame is None:
            return DetectionResult([], [], [], [])

        try:
            started = time.perf_counter()
            frame = self._preprocess(frame)
            with self.lock:
                kwargs = dict(
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self._imgsz,
                    max_det=self._max_det,
                    verbose=False
                )
                # NCNN runs its own ARM-optimised backend and takes no device argument.
                if not self.use_ncnn:
                    kwargs['device'] = self._device
                if self.weevil_class_ids:
                    kwargs['classes'] = self.weevil_class_ids
                results = self.model(frame, **kwargs)
            inference_ms = (time.perf_counter() - started) * 1000

            boxes = []
            confidences = []
            class_ids = []

            # A generic COCO checkpoint has no weevil class, so report nothing rather than
            # counting people and cars as rice weevils.
            if not self.is_generic_model:
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].cpu().numpy())
                            if self.weevil_class_ids is not None and cls_id not in self.weevil_class_ids:
                                continue
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append([int(x1), int(y1), int(x2), int(y2)])
                            confidences.append(float(box.conf[0].cpu().numpy()))
                            class_ids.append(cls_id)

            self._record_inference_time(inference_ms)
            return DetectionResult(boxes, confidences, class_ids, self.class_names, inference_ms)
        except Exception as e:
            print(f"Detection error: {e}")
            return DetectionResult([], [], [], [])

    def _record_inference_time(self, inference_ms: float):
        self.last_inference_ms = inference_ms
        self._inference_samples += 1
        # Exponential moving average keeps this O(1) and stable on the Pi.
        weight = 0.1 if self._inference_samples > 1 else 1.0
        self.avg_inference_ms = (1 - weight) * self.avg_inference_ms + weight * inference_ms

    def get_model_info(self) -> Dict[str, object]:
        """Model details for display in the UI."""
        return {
            'model_name': os.path.basename(self.loaded_path),
            'architecture': 'YOLOv11n',
            'backend': self.backend,
            'imgsz': self._imgsz,
            'trained_imgsz': self._trained_imgsz,
            'max_det': self._max_det,
            'device': 'Raspberry Pi 5 CPU' if self._is_pi else 'CPU',
            'classes': self.class_names,
            'weevil_class_ids': self.weevil_class_ids,
            'clahe': self.use_clahe,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'initialized': self.initialized,
            'warning': self.warning,
            'metrics': dict(TRAINED_METRICS),
        }

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
        if self.weevil_class_ids:
            return sum(1 for cls_id in detection.class_ids if cls_id in self.weevil_class_ids)
        return detection.count
