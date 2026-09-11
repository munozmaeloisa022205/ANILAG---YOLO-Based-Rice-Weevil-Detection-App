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


def _configure_cpu_threads(is_pi: bool) -> Optional[int]:
    """Cap intra-op parallelism so inference does not starve the rest of the app.

    The Raspberry Pi 5 has four Cortex-A76 cores. PyTorch and OpenCV each default
    to using all of them, so a detection cycle leaves nothing for the Qt GUI
    thread, the two camera threads and the SQLite writes - the interface visibly
    freezes for the length of every inference. Reserving one core keeps the UI
    responsive at a small cost in per-frame latency.

    Override with THREAD_COUNT; 0 means "leave the library defaults alone".
    """
    cores = os.cpu_count() or 1
    # On Pi 5: reserve one core for GUI/camera/DB threads (4 cores → 3).
    # On Windows/desktop: cap at 4 to avoid thread contention. With 20 cores
    # the library default saturates the scheduler and starves the Qt event loop,
    # camera capture threads and video encoding — the GUI freezes and camera
    # FPS drops. 4 inference threads is enough for YOLOv11n and leaves plenty
    # of headroom for the rest of the app.
    default = max(1, cores - 1) if is_pi else min(4, cores)
    try:
        threads = int(os.getenv('THREAD_COUNT', str(default)))
    except ValueError:
        threads = default
    if threads <= 0:
        return None

    try:
        import torch
        torch.set_num_threads(threads)
    except Exception as e:
        print(f"Could not set torch thread count: {e}")
    try:
        cv2.setNumThreads(threads)
    except Exception as e:
        print(f"Could not set OpenCV thread count: {e}")
    return threads


def _parse_tile_grid(value: str) -> Tuple[int, int]:
    """Parse YOLO_TILES ("2x2", "3x2", "off", "1") into (columns, rows).

    Anything that means a single tile disables tiling, since a 1x1 grid is just
    the full-frame pass that always runs.
    """
    text = (value or '').strip().lower()
    if text in ('', 'off', 'none', 'false', '0', '1', '1x1'):
        return (1, 1)
    try:
        if 'x' in text:
            cols, rows = text.split('x', 1)
            return (max(1, int(cols)), max(1, int(rows)))
        size = max(1, int(text))
        return (size, size)
    except ValueError:
        print(f"Invalid YOLO_TILES value {value!r}; tiling disabled")
        return (1, 1)


def _merge_boxes(boxes: List[List[int]], scores: List[float],
                 iou_threshold: float, containment_threshold: float) -> List[int]:
    """Deduplicate boxes from the full-frame and tiled passes.

    Returns the indices to keep, highest score first.

    Two boxes are considered duplicates (the same weevil seen in overlapping
    tiles) when their CENTROIDS are close together relative to the box size.
    This is more precise than IoU/containment alone: clustered weevils that
    partially overlap still have distinct centroids, so they survive as
    separate counts. A high-IoU fallback catches near-identical boxes (the
    same object detected twice in the same tile region), and a very high
    containment fallback catches fragments fully inside the complete box.
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    areas = [max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in boxes]
    centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
    diagonals = [((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5 for b in boxes]
    kept: List[int] = []
    while order:
        current = order.pop(0)
        kept.append(current)
        cx1, cy1, cx2, cy2 = boxes[current]
        ccx, ccy = centroids[current]
        cd = diagonals[current]
        remaining = []
        for other in order:
            ox1, oy1, ox2, oy2 = boxes[other]
            # Centroid distance: duplicates from overlapping tiles have
            # centroids within a fraction of the box diagonal. Two distinct
            # weevils, even clustered, have centroids at least one body-width
            # apart — well beyond this threshold.
            # Reduced from 0.5 to 0.3 to avoid merging distinct clustered
            # weevils on the left camera (false negatives).
            ocx, ocy = centroids[other]
            od = diagonals[other]
            centroid_dist = ((ccx - ocx) ** 2 + (ccy - ocy) ** 2) ** 0.5
            min_diag = min(cd, od)
            if min_diag > 0 and centroid_dist < min_diag * 0.3:
                continue  # same object — suppress the lower-score duplicate
            iw = min(cx2, ox2) - max(cx1, ox1)
            ih = min(cy2, oy2) - max(cy1, oy1)
            if iw <= 0 or ih <= 0:
                remaining.append(other)
                continue
            intersection = iw * ih
            union = areas[current] + areas[other] - intersection
            iou = intersection / union if union > 0 else 0.0
            smaller = min(areas[current], areas[other])
            containment = intersection / smaller if smaller > 0 else 0.0
            # Only merge on very high overlap — near-identical boxes or a
            # fragment fully inside the complete box. The thresholds here are
            # intentionally high so distinct weevils that merely partially
            # overlap are NOT merged.
            if iou > iou_threshold or containment > containment_threshold:
                continue
            remaining.append(other)
        order = remaining
    return kept


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
        self._threads = _configure_cpu_threads(self._is_pi)
        # Must match the imgsz the weights were trained at (640 for sitophilus_oryzae_v2-3)
        # or recall on small weevils drops. Lower only if the Pi 5 cannot keep up.
        self._imgsz = int(os.getenv('YOLO_IMGSZ', str(TRAINED_IMGSZ)))
        self._trained_imgsz = int(os.getenv('MODEL_TRAINED_IMGSZ', str(TRAINED_IMGSZ)))
        self._max_det = int(os.getenv('YOLO_MAX_DET', '300'))  # matches the training/val default
        self.use_ncnn = False
        self.loaded_path = model_path
        # Test-time augmentation: multi-scale + flip inference. Improves detection
        # of small/distant weevils by running the model at multiple scales and
        # merging results. Costs ~3x inference time but dramatically improves
        # recall on small objects.
        self._augment = os.getenv('YOLO_AUGMENT', 'true').lower() in ('true', '1', 'yes', 'on')
        # Tiled ("sliced") inference for distant subjects.
        #
        # The model input is fixed at 640px, so a 1920x1080 frame is downscaled
        # 3x before inference. A weevil about a foot from the lens is only ~30px
        # wide in the full frame, which becomes ~10px after that downscale -
        # below what the detector can resolve. Measured recall on real dataset
        # images pasted into a 1080p canvas confirms the cliff: ~97% when the
        # subject fills the frame, but 1.8% once it covers only a quarter of it,
        # and 0% beyond that. NCNN and PyTorch fail identically, so this is a
        # scale limit, not a conversion artefact.
        #
        # Splitting the frame into overlapping tiles and running the model on
        # each tile keeps the subject near its native pixel size, restoring
        # recall at distance. Cost is one inference per tile plus one full-frame
        # pass (which still catches close-up weevils spanning several tiles).
        # Measured recall when the sample fills a quarter of the frame:
        # off 1.8%, 2x2 61.1%, 3x3 89.4%, 4x4 94.7%. 3x3 is the default as the
        # smallest grid that holds up at about a foot of camera distance.
        self._tile_grid = _parse_tile_grid(os.getenv('YOLO_TILES', '3x3'))
        self._tile_overlap = float(os.getenv('YOLO_TILE_OVERLAP', '0.2'))
        # Thresholds for merging the per-tile results back together. The merge
        # uses centroid distance as the primary deduplication criterion (same
        # weevil seen in overlapping tiles has near-identical centroids), with
        # high-IoU and high-containment fallbacks for near-identical boxes and
        # tile-seam fragments. These thresholds are intentionally HIGH so that
        # distinct weevils that cluster together (common in grain) are NOT
        # merged — only near-duplicates are suppressed.
        self._merge_iou = float(os.getenv('YOLO_TILE_MERGE_IOU', '0.85'))
        self._merge_containment = float(os.getenv('YOLO_TILE_MERGE_CONTAINMENT', '0.9'))
        # Post-processing size filters to reduce false positives. Weevils are
        # small insects — boxes covering a large fraction of the frame are likely
        # background objects, and tiny boxes (< 100 px²) are noise.
        # Defaults measured from the 46,763 training boxes by
        # tools/analyze_dataset_boxes.py. Each rejects <2% of real weevils.
        # Tightening them by guesswork silently destroys valid detections: the
        # dataset's median weevil covers 13% of the frame, so a "weevils are
        # small" assumption (e.g. 5% cap) discards ~65% of true positives.
        self._min_box_area = int(os.getenv('YOLO_MIN_BOX_AREA', '100'))
        self._max_box_area_ratio = float(os.getenv('YOLO_MAX_BOX_AREA_RATIO', '0.85'))
        self._max_aspect_ratio = float(os.getenv('YOLO_MAX_ASPECT_RATIO', '20'))
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
        self._detect_calls = 0  # diagnostic logging counter

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
                print(f"Inference threads: {self._threads or 'library default'} "
                      f"of {os.cpu_count()} cores")
                print(f"Inference image size: {self._imgsz} (trained at {self._trained_imgsz})")
                print(f"Classes: {self.class_names}")
                print(f"Confidence threshold: {self.confidence_threshold}")
                print(f"Augment (TTA): {'enabled' if (self._augment and not self.use_ncnn) else 'disabled'}"
                      f"{' (NCNN does not support TTA)' if (self._augment and self.use_ncnn) else ''}")
                print(f"Box filters: min_area={self._min_box_area}px² "
                      f"max_ratio={self._max_box_area_ratio} "
                      f"max_aspect={self._max_aspect_ratio}")
                cols, rows = self._tile_grid
                if (cols, rows) == (1, 1):
                    print("Tiled inference: off (close range only)")
                else:
                    print(f"Tiled inference: {cols}x{rows} grid, "
                          f"{self._tile_overlap:.0%} overlap "
                          f"({cols * rows} inferences/frame)")
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

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Public accessor for the CLAHE preprocessing applied before inference.

        Returns the exact image the model sees (adaptive-equalized lightness
        channel, colour preserved, geometry unchanged). Callers can store this
        alongside the annotated raw frame to audit what the detector actually
        received.
        """
        return self._preprocess(frame)

    def _infer(self, frame: np.ndarray) -> Tuple[List[List[int]], List[float], List[int]]:
        """Run one inference pass and apply the size/aspect filters.

        Coordinates are relative to the frame passed in, so a caller working on a
        tile must offset them back into full-frame space.
        """
        with self.lock:
            # NCNN supports: conf, iou, max_det, verbose (post-processing params).
            # NCNN does NOT support: augment (TTA), device, imgsz, classes.
            # - augment: TTA is PyTorch-only; NCNN runs a fixed graph.
            # - imgsz: baked into the NCNN export at conversion time.
            # - device: NCNN manages its own threading.
            # - classes: unnecessary for single-class models; can cause NCNN issues.
            kwargs = dict(
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self._max_det,
                verbose=False
            )
            if not self.use_ncnn:
                # PyTorch: full parameter set including TTA and device.
                kwargs['device'] = self._device
                kwargs['imgsz'] = self._imgsz
                kwargs['augment'] = self._augment
                if self.weevil_class_ids:
                    kwargs['classes'] = self.weevil_class_ids
            results = self.model(frame, **kwargs)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []
        # A generic COCO checkpoint has no weevil class, so report nothing rather
        # than counting people and cars as rice weevils.
        if self.is_generic_model:
            return boxes, confidences, class_ids

        frame_h, frame_w = frame.shape[:2]
        max_box_area = frame_h * frame_w * self._max_box_area_ratio
        raw_count = 0
        rejected_class = 0
        rejected_size = 0
        rejected_aspect = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                raw_count += 1
                cls_id = int(box.cls[0].cpu().numpy())
                if self.weevil_class_ids is not None and cls_id not in self.weevil_class_ids:
                    rejected_class += 1
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                # Reject boxes too small to be an insect, or so large they are
                # clearly background rather than a weevil.
                if box_area < self._min_box_area or box_area > max_box_area:
                    rejected_size += 1
                    continue
                # Reject extreme aspect ratios (table edges, shadow streaks).
                if box_w > 0 and box_h > 0:
                    if max(box_w, box_h) / min(box_w, box_h) > self._max_aspect_ratio:
                        rejected_aspect += 1
                        continue
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(box.conf[0].cpu().numpy()))
                class_ids.append(cls_id)
        # Log when raw detections exist but all are filtered out — this catches
        # a too-aggressive size/aspect filter silently eating every detection.
        if raw_count > 0 and len(boxes) == 0 and self._detect_calls % 30 == 0:
            print(f"[INFER] raw={raw_count} kept={len(boxes)} "
                  f"rej_class={rejected_class} rej_size={rejected_size} "
                  f"rej_aspect={rejected_aspect} "
                  f"frame={frame_w}x{frame_h} max_area={max_box_area} "
                  f"min_area={self._min_box_area}")
        return boxes, confidences, class_ids

    def _tiles(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """Overlapping tile rectangles as (x0, y0, x1, y1) in frame coordinates.

        Tiles overlap so a weevil sitting on a seam is still wholly inside at
        least one tile; the NMS merge afterwards removes the duplicates.
        """
        cols, rows = self._tile_grid
        if cols <= 1 and rows <= 1:
            return []
        overlap = min(0.9, max(0.0, self._tile_overlap))
        tile_w = int(width / cols)
        tile_h = int(height / rows)
        pad_x = int(tile_w * overlap / 2)
        pad_y = int(tile_h * overlap / 2)
        rects = []
        for row in range(rows):
            for col in range(cols):
                x0 = max(0, col * tile_w - pad_x)
                y0 = max(0, row * tile_h - pad_y)
                x1 = min(width, (col + 1) * tile_w + pad_x)
                y1 = min(height, (row + 1) * tile_h + pad_y)
                if x1 - x0 >= 32 and y1 - y0 >= 32:
                    rects.append((x0, y0, x1, y1))
        return rects

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self.initialized or self.model is None or frame is None:
            return DetectionResult([], [], [], [])

        try:
            started = time.perf_counter()
            frame = self._preprocess(frame)
            height, width = frame.shape[:2]

            # Full-frame pass. Catches close-up weevils that span several tiles.
            # Skipped when tiling is enabled — our weevils are 9-23px, far too
            # small to span tiles (each tile is ~960x540). The full-frame pass
            # adds ~100ms per camera with zero added recall, so skipping it
            # cuts cycle time by ~20% (from 10 passes to 8 per cycle).
            cols, rows = self._tile_grid
            if cols <= 1 and rows <= 1:
                boxes, confidences, class_ids = self._infer(frame)
                full_frame_count = len(boxes)
            else:
                boxes, confidences, class_ids = [], [], []
                full_frame_count = 0

            # Tiled passes. Each tile is inferred at its own scale, so a subject
            # only a few pixels wide in the downscaled full frame is large enough
            # to detect here. Coordinates are shifted back into full-frame space.
            tile_count = 0
            for x0, y0, x1, y1 in self._tiles(width, height):
                t_boxes, t_conf, t_cls = self._infer(frame[y0:y1, x0:x1])
                tile_count += len(t_boxes)
                for (bx1, by1, bx2, by2), conf, cls_id in zip(t_boxes, t_conf, t_cls):
                    boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
                    confidences.append(conf)
                    class_ids.append(cls_id)

            pre_merge_count = len(boxes)

            # Overlapping tiles report the same insect more than once, so merge
            # before counting. Not needed without tiling, since the model already
            # ran NMS internally on the single pass.
            if self._tile_grid != (1, 1) and len(boxes) > 1:
                keep = _merge_boxes(boxes, confidences,
                                    self._merge_iou, self._merge_containment)
                boxes = [boxes[i] for i in keep]
                confidences = [confidences[i] for i in keep]
                class_ids = [class_ids[i] for i in keep]

            # Diagnostic logging: shows up in journalctl -u anilag.service.
            # Prints every 30th detection cycle to avoid flooding the log.
            self._detect_calls += 1
            if self._detect_calls % 30 == 0:
                print(f"[DETECT] full={full_frame_count} tiles={tile_count} "
                      f"pre_merge={pre_merge_count} post_merge={len(boxes)} "
                      f"conf={self.confidence_threshold} iou={self.iou_threshold} "
                      f"merge_iou={self._merge_iou} merge_cont={self._merge_containment} "
                      f"clahe={self.use_clahe} ncnn={self.use_ncnn} "
                      f"imgsz={self._imgsz} tiles={self._tile_grid} "
                      f"inference_ms={self.avg_inference_ms:.1f}")

            inference_ms = (time.perf_counter() - started) * 1000
            self._record_inference_time(inference_ms)
            return DetectionResult(boxes, confidences, class_ids, self.class_names, inference_ms)
        except Exception as e:
            print(f"[DETECT ERROR] {e}")
            import traceback
            traceback.print_exc()
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
            'threads': self._threads,
            'classes': self.class_names,
            'weevil_class_ids': self.weevil_class_ids,
            'clahe': self.use_clahe,
            # NCNN runs a fixed graph, so TTA is only ever active on PyTorch.
            'augment': self._augment and not self.use_ncnn,
            'tile_grid': f"{self._tile_grid[0]}x{self._tile_grid[1]}",
            'tile_passes': self._tile_grid[0] * self._tile_grid[1] if (self._tile_grid[0] > 1 or self._tile_grid[1] > 1) else 1,
            'min_box_area': self._min_box_area,
            'max_box_area_ratio': self._max_box_area_ratio,
            'max_aspect_ratio': self._max_aspect_ratio,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'initialized': self.initialized,
            'warning': self.warning,
            'metrics': dict(TRAINED_METRICS),
        }

    def draw_detections(self, frame: np.ndarray, detection: DetectionResult, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        annotated_frame = frame

        for i, (box, conf, cls_id) in enumerate(zip(detection.boxes, detection.confidences, detection.class_ids)):
            x1, y1, x2, y2 = box
            class_name = detection.class_names[cls_id] if cls_id < len(detection.class_names) else f"Class {cls_id}"

            # Draw bounding box — thick (3px) so it's visible on small screens.
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

            # Draw label with background for readability
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
