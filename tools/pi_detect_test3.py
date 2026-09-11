#!/usr/bin/env python3
"""SSH into the Pi and run a direct detection test on stored images."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=240):
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

    # Find the venv Python
    run(client, "ls ~/anilag/.venv/bin/python3 2>/dev/null; which python3; pip3 list 2>/dev/null | grep -iE 'opencv|ultralytics|ncnn|torch' | head -10", "Find Python/venv")

    # Write a test script to a temp file on the Pi
    test_script = r'''import os, sys, time, glob
import numpy as np
os.chdir(os.path.expanduser("~/anilag"))
from pathlib import Path
env_file = Path("config.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, "src")
import cv2
from detection.yolov11_detector import YOLOv11Detector

detector = YOLOv11Detector()
if not detector.initialize():
    print("FAILED to initialize detector")
    sys.exit(1)

print("Backend:", detector.backend)
print("NCNN:", detector.use_ncnn)
print("CLAHE:", detector.use_clahe)
print("Confidence:", detector.confidence_threshold)
print("IOU:", detector.iou_threshold)
print("Tiles:", detector._tile_grid)
print("Merge IoU:", detector._merge_iou)
print("Merge Containment:", detector._merge_containment)
print("Class names:", detector.class_names)
print("Weevil class IDs:", detector.weevil_class_ids)
print("Min box area:", detector._min_box_area)
print("Max box area ratio:", detector._max_box_area_ratio)
print("Max aspect ratio:", detector._max_aspect_ratio)

# Test 1: Random noise frame
print("\n--- Test 1: Random noise frame ---")
np.random.seed(42)
frame1 = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
t0 = time.perf_counter()
result1 = detector.detect(frame1)
t1 = time.perf_counter()
print("Count:", result1.count)
print("Inference ms:", round((t1-t0)*1000, 1))

# Test 2: Find stored scan images and run detection on them
print("\n--- Test 2: Detection on stored scan images ---")
scan_dirs = sorted(glob.glob("previous_scans/scan_*"))
print("Found", len(scan_dirs), "scan directories")

test_images = []
for d in scan_dirs[-2:]:
    imgs = sorted(glob.glob(os.path.join(d, "detected_images", "*.jpg")))
    annotated_left = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
    if annotated_left:
        test_images.extend(annotated_left[-2:])

if test_images:
    for test_img in test_images[:3]:
        print("\n  Testing:", os.path.basename(test_img))
        img = cv2.imread(test_img)
        if img is not None:
            print("  Image shape:", img.shape)
            # Run at default confidence (0.5)
            t0 = time.perf_counter()
            result = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count (conf=0.5):", result.count)
            if result.boxes:
                print("  Boxes:", result.boxes[:5])
                confs = ["%.3f" % c for c in result.confidences[:5]]
                print("  Confidences:", confs)
            print("  Inference ms:", round((t1-t0)*1000, 1))

            # Run at lower confidence (0.25)
            detector.confidence_threshold = 0.25
            t0 = time.perf_counter()
            result2 = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count (conf=0.25):", result2.count)
            if result2.boxes:
                confs2 = ["%.3f" % c for c in result2.confidences[:10]]
                print("  Confidences:", confs2)
            print("  Inference ms:", round((t1-t0)*1000, 1))

            # Run at very low confidence (0.1)
            detector.confidence_threshold = 0.1
            t0 = time.perf_counter()
            result3 = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count (conf=0.1):", result3.count)
            if result3.boxes:
                confs3 = ["%.3f" % c for c in result3.confidences[:10]]
                print("  Confidences:", confs3)
            print("  Inference ms:", round((t1-t0)*1000, 1))

            # Reset
            detector.confidence_threshold = 0.5
else:
    print("  No stored images found to test")

# Test 3: Check if the raw (non-annotated) frames are stored
print("\n--- Test 3: Check for raw frames ---")
for d in scan_dirs[-1:]:
    all_imgs = sorted(glob.glob(os.path.join(d, "detected_images", "*.jpg")))
    print("  All images in", os.path.basename(d) + ":")
    for f in all_imgs[:10]:
        print("   ", os.path.basename(f))

print("\nDetect calls:", detector._detect_calls)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_detect_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    # Try venv python first, fall back to system python
    run(client, "~/anilag/.venv/bin/python3 /tmp/pi_detect_test.py 2>&1 || python3 /tmp/pi_detect_test.py 2>&1", "Direct detection test", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
