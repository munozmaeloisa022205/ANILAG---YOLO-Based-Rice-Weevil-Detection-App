"""Verify the NCNN export detects the same weevils as the PyTorch weights.

Answers two deployment questions:
  1. Did the NCNN conversion for the Pi 5 degrade detection?
  2. Does the model still detect weevils that are far from the camera (small in
     frame), as some dataset images show?

Runs both backends over real dataset images at full size and at progressively
smaller scales, which simulates moving the camera away from the sample.
Requires the AI-MODEL dataset directory to be present.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(
    ROOT, 'AI-MODEL',
    'Sitophilus Oryzae YOLO Detection.v2-sitophilus_oryzae_augmented_dataset_v1.0_yolov11n.yolov11',
)
PT_MODEL = os.path.join(ROOT, 'models', 'sitophilus_oryzae_v2-3_best.pt')
NCNN_MODEL = os.path.join(ROOT, 'models', 'sitophilus_oryzae_v2-3_best_ncnn_model')

# Fractions of a 1920x1080 frame the weevil is pasted into. 1.0 keeps the
# original close-up framing; smaller values emulate a camera further away.
SCALES = (1.0, 0.5, 0.25, 0.12, 0.06)


def load_samples(limit=12):
    """Return (image, label_count) pairs from the validation split."""
    samples = []
    for image_path in sorted(glob.glob(os.path.join(DATASET, 'valid', 'images', '*.jpg')))[:limit]:
        label_path = image_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
        label_path = os.path.splitext(label_path)[0] + '.txt'
        if not os.path.exists(label_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        with open(label_path) as handle:
            count = sum(1 for line in handle if len(line.split()) >= 5)
        samples.append((os.path.basename(image_path), image, count))
    return samples


def embed(image, scale, canvas=(1080, 1920)):
    """Paste the 640x640 sample into a 1080p canvas at the given scale.

    This reproduces what the camera actually delivers: a small subject inside a
    large frame, rather than a tightly cropped macro shot.
    """
    if scale >= 1.0:
        return cv2.resize(image, (canvas[1], canvas[0]), interpolation=cv2.INTER_LINEAR)
    height = max(8, int(canvas[0] * scale))
    width = max(8, int(canvas[1] * scale))
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    # Mid-grey surround: neutral, and avoids a pure-black border that the
    # blank-frame heuristics elsewhere would treat as signal loss.
    frame = np.full((canvas[0], canvas[1], 3), 128, dtype=np.uint8)
    y0 = (canvas[0] - height) // 2
    x0 = (canvas[1] - width) // 2
    frame[y0:y0 + height, x0:x0 + width] = small
    return frame


def build(model_path, use_ncnn, tiles='off'):
    from src.detection.yolov11_detector import YOLOv11Detector
    os.environ['USE_NCNN'] = 'true' if use_ncnn else 'false'
    os.environ['YOLO_TILES'] = tiles
    detector = YOLOv11Detector(
        model_path=model_path,
        confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
        iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.7')),
    )
    if not detector.initialize():
        return None
    if detector.use_ncnn != use_ncnn:
        print(f"  (requested use_ncnn={use_ncnn}, got {detector.use_ncnn})")
    return detector


def evaluate(detector, samples):
    """Return {scale: (detected, expected)} totals."""
    totals = {}
    for scale in SCALES:
        detected = expected = 0
        for _, image, count in samples:
            frame = embed(image, scale)
            result = detector.detect(frame)
            detected += result.count
            expected += count
        totals[scale] = (detected, expected)
    return totals


def main():
    if not os.path.isdir(DATASET):
        print(f"Dataset not found: {DATASET}")
        print("This check needs the AI-MODEL dataset; skipping.")
        return 0

    samples = load_samples()
    if not samples:
        print("No validation images found.")
        return 1
    print(f"Loaded {len(samples)} validation images "
          f"({sum(c for _, _, c in samples)} annotated weevils)\n")

    results = {}
    timings = {}
    for label, use_ncnn, tiles in (
        ('PyTorch', False, 'off'),
        ('NCNN', True, 'off'),
        ('NCNN+2x2', True, '2x2'),
    ):
        print(f"--- {label} ---")
        detector = build(PT_MODEL, use_ncnn, tiles)
        if detector is None:
            print("  could not initialise; skipping\n")
            continue
        print(f"  backend={detector.backend} clahe={detector.use_clahe} "
              f"conf={detector.confidence_threshold} tiles={detector._tile_grid}")
        results[label] = evaluate(detector, samples)
        timings[label] = detector.avg_inference_ms
        print(f"  avg {detector.avg_inference_ms:.0f} ms/frame\n")

    print("=== Recall by subject scale (detected / annotated) ===")
    print("scale = fraction of a 1920x1080 frame the sample occupies")
    header = "  " + "scale".ljust(8) + "".join(k.ljust(22) for k in results)
    print(header)
    for scale in SCALES:
        row = f"  {scale:<8.2f}"
        for label in results:
            detected, expected = results[label][scale]
            pct = (100.0 * detected / expected) if expected else 0.0
            row += f"{detected}/{expected} ({pct:5.1f}%)".ljust(22)
        print(row)

    if 'PyTorch' in results and 'NCNN' in results:
        print("\n=== Did the NCNN conversion cost accuracy? ===")
        for scale in SCALES:
            pt = results['PyTorch'][scale][0]
            nc = results['NCNN'][scale][0]
            if pt == 0 and nc == 0:
                verdict = "both found nothing"
            elif pt == 0:
                verdict = "NCNN found more"
            else:
                verdict = f"{100.0 * (nc - pt) / pt:+.1f}% vs PyTorch"
            print(f"  scale {scale:<5.2f}: PyTorch={pt:<5} NCNN={nc:<5} {verdict}")

    if 'NCNN' in results and 'NCNN+2x2' in results:
        print("\n=== Does tiling recover distant weevils? ===")
        for scale in SCALES:
            plain, expected = results['NCNN'][scale]
            tiled = results['NCNN+2x2'][scale][0]
            p_pct = (100.0 * plain / expected) if expected else 0.0
            t_pct = (100.0 * tiled / expected) if expected else 0.0
            print(f"  scale {scale:<5.2f}: full-frame {plain:>3}/{expected} ({p_pct:5.1f}%) "
                  f"-> tiled {tiled:>3}/{expected} ({t_pct:5.1f}%)")
        print(f"\n  cost: {timings['NCNN']:.0f} ms -> {timings['NCNN+2x2']:.0f} ms per frame "
              f"on this machine")
    return 0


if __name__ == '__main__':
    sys.exit(main())
