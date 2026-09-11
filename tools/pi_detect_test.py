#!/usr/bin/env python3
"""SSH into the Pi and run a direct detection test on a synthetic frame."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=120):
    if label:
        print(f"\n=== {label} ===")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(f"[stderr] {err.rstrip()}")
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Run a direct detection test: create a test frame, run detect(), print results
    test_script = r'''
import os, sys, time
os.chdir(os.path.expanduser("~/anilag"))
# Load env
from pathlib import Path
env_file = Path("config.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, "src")
import numpy as np
from detection.yolov11_detector import YOLOv11Detector

detector = YOLOv11Detector()
if not detector.initialize():
    print("FAILED to initialize detector")
    sys.exit(1)

print(f"Backend: {detector.backend}")
print(f"NCNN: {detector.use_ncnn}")
print(f"CLAHE: {detector.use_clahe}")
print(f"Confidence: {detector.confidence_threshold}")
print(f"IOU: {detector.iou_threshold}")
print(f"Tiles: {detector._tile_grid}")
print(f"Merge IoU: {detector._merge_iou}")
print(f"Merge Containment: {detector._merge_containment}")
print(f"Class names: {detector.class_names}")
print(f"Weevil class IDs: {detector.weevil_class_ids}")
print(f"Min box area: {detector._min_box_area}")
print(f"Max box area ratio: {detector._max_box_area_ratio}")
print(f"Max aspect ratio: {detector._max_aspect_ratio}")

# Test 1: Random noise frame (should find 0 or very few detections)
print("\n--- Test 1: Random noise frame ---")
np.random.seed(42)
frame1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
t0 = time.perf_counter()
result1 = detector.detect(frame1)
t1 = time.perf_counter()
print(f"Count: {result1.count}")
print(f"Boxes: {result1.boxes[:5]}")
print(f"Confidences: {result1.confidences[:5]}")
print(f"Inference ms: {(t1-t0)*1000:.1f}")

# Test 2: Solid color frame (should find 0 detections)
print("\n--- Test 2: Solid color frame ---")
frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
frame2[:] = (100, 100, 100)  # gray
t0 = time.perf_counter()
result2 = detector.detect(frame2)
t1 = time.perf_counter()
print(f"Count: {result2.count}")
print(f"Inference ms: {(t1-t0)*1000:.1f}")

# Test 3: Check if there are any stored scan images we can test on
print("\n--- Test 3: Check for stored scan images ---")
import glob
scan_dirs = sorted(glob.glob("previous_scans/scans_*"))
if not scan_dirs:
    scan_dirs = sorted(glob.glob("previous_scans/scan_*"))
print(f"Found {len(scan_dirs)} scan directories")
for d in scan_dirs[-3:]:
    imgs = glob.glob(os.path.join(d, "detected_images", "*.jpg"))
    print(f"  {d}: {len(imgs)} images")
    for img in imgs[:2]:
        print(f"    {os.path.basename(img)}")

# Test 4: If we have stored images, run detection on one
if scan_dirs:
    for d in scan_dirs[-3:]:
        imgs = glob.glob(os.path.join(d, "detected_images", "*.jpg"))
        if imgs:
            # Pick an annotated image (not preprocessed) from the left camera
            annotated = [f for f in imgs if "left_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
            if annotated:
                test_img = annotated[0]
                print(f"\n--- Test 4: Detection on stored image: {os.path.basename(test_img)} ---")
                import cv2
                img = cv2.imread(test_img)
                if img is not None:
                    print(f"Image shape: {img.shape}")
                    t0 = time.perf_counter()
                    result = detector.detect(img)
                    t1 = time.perf_counter()
                    print(f"Count: {result.count}")
                    print(f"Boxes: {result.boxes[:10]}")
                    print(f"Confidences: {[f'{c:.3f}' for c in result.confidences[:10]]}")
                    print(f"Inference ms: {(t1-t0)*1000:.1f}")
                break

print("\n--- Diagnostic counter ---")
print(f"Detect calls: {detector._detect_calls}")

'''

    run(client, f"cd ~/anilag && python3 -c '{test_script}'", "Direct detection test", timeout=180)

    # Also check recent journalctl for any DETECT output during the last scan
    run(client, "sudo journalctl -u anilag.service --since '1 hour ago' --no-pager 2>/dev/null | grep -E 'DETECT|INFER|ERROR|Count|detect' | tail -30", "Recent detection logs (last hour)")

    # Check for any scan images stored recently
    run(client, "find ~/anilag/previous_scans -name '*.jpg' -newer ~/anilag/config.env 2>/dev/null | head -20", "Recent stored images")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
