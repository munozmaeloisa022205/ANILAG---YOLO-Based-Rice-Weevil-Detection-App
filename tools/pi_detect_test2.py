#!/usr/bin/env python3
"""SSH into the Pi and run a direct detection test on stored images."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=180):
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

    # Write a test script to a temp file on the Pi, then run it
    test_script = r'''import os, sys, time, glob, cv2
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

for d in scan_dirs[-2:]:
    imgs = sorted(glob.glob(os.path.join(d, "detected_images", "*.jpg")))
    # Filter to non-preprocessed left images
    annotated_left = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
    if annotated_left:
        test_img = annotated_left[-1]  # most recent
        print("\n  Testing:", os.path.basename(test_img))
        img = cv2.imread(test_img)
        if img is not None:
            print("  Image shape:", img.shape)
            t0 = time.perf_counter()
            result = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count:", result.count)
            print("  Boxes:", result.boxes[:10])
            confs = ["%.3f" % c for c in result.confidences[:10]]
            print("  Confidences:", confs)
            print("  Inference ms:", round((t1-t0)*1000, 1))
    break

# Test 3: Try with lower confidence threshold
print("\n--- Test 3: Detection with conf=0.25 (lower threshold) ---")
detector.confidence_threshold = 0.25
if scan_dirs:
    imgs = sorted(glob.glob(os.path.join(scan_dirs[-1], "detected_images", "*.jpg")))
    annotated_left = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
    if annotated_left:
        test_img = annotated_left[-1]
        print("  Testing:", os.path.basename(test_img))
        img = cv2.imread(test_img)
        if img is not None:
            t0 = time.perf_counter()
            result = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count:", result.count)
            print("  Boxes:", result.boxes[:10])
            confs = ["%.3f" % c for c in result.confidences[:10]]
            print("  Confidences:", confs)
            print("  Inference ms:", round((t1-t0)*1000, 1))

# Test 4: Try with conf=0.1
print("\n--- Test 4: Detection with conf=0.1 (very low threshold) ---")
detector.confidence_threshold = 0.1
if scan_dirs:
    imgs = sorted(glob.glob(os.path.join(scan_dirs[-1], "detected_images", "*.jpg")))
    annotated_left = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
    if annotated_left:
        test_img = annotated_left[-1]
        print("  Testing:", os.path.basename(test_img))
        img = cv2.imread(test_img)
        if img is not None:
            t0 = time.perf_counter()
            result = detector.detect(img)
            t1 = time.perf_counter()
            print("  Count:", result.count)
            print("  Boxes:", result.boxes[:10])
            confs = ["%.3f" % c for c in result.confidences[:10]]
            print("  Confidences:", confs)
            print("  Inference ms:", round((t1-t0)*1000, 1))

print("\nDetect calls:", detector._detect_calls)
'''

    # Write the script to a temp file on the Pi
    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_detect_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "python3 /tmp/pi_detect_test.py", "Direct detection test", timeout=300)

    # Also check the database for detection counts
    run(client, "cd ~/anilag && python3 -c \"import sys; sys.path.insert(0,'src'); from backend.database import DatabaseManager; db=DatabaseManager(); scans=db.get_all_scans(); print('Scans:', len(scans)); [print(f'  {s[\"scan_id\"]}: max_count={s.get(\"max_count\",\"?\")} images={s.get(\"image_count\",\"?\")}') for s in scans[-5:]]\" 2>&1", "Database scan summary")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
