#!/usr/bin/env python3
"""SSH into the Pi and run detection test using the venv Python like the service does."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=280):
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

    # Check for local logging.py that might shadow stdlib
    run(client, "find ~/anilag/src -name 'logging.py' -o -name 'logging' -type d 2>/dev/null; find ~/anilag -maxdepth 1 -name 'logging.py' 2>/dev/null", "Check for local logging.py")

    # Check how main.py imports things
    run(client, "head -30 ~/anilag/main.py", "main.py imports")

    # Write a test script that runs from ~/anilag using the venv
    test_script = r'''import os, sys, time, glob
os.chdir(os.path.expanduser("~/anilag"))
# Load env vars
from pathlib import Path
env_file = Path("config.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Import the same way main.py does
import numpy as np
import cv2
from src.detection.yolov11_detector import YOLOv11Detector

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
    annotated_left = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
    if annotated_left:
        test_img = annotated_left[-1]
        print("\n  Testing:", os.path.basename(test_img))
        img = cv2.imread(test_img)
        if img is not None:
            print("  Image shape:", img.shape)
            for conf_val in [0.5, 0.25, 0.1]:
                detector.confidence_threshold = conf_val
                t0 = time.perf_counter()
                result = detector.detect(img)
                t1 = time.perf_counter()
                print("  Count (conf=" + str(conf_val) + "):", result.count)
                if result.boxes:
                    print("  Boxes:", result.boxes[:5])
                    confs = ["%.3f" % c for c in result.confidences[:5]]
                    print("  Confidences:", confs)
                print("  Inference ms:", round((t1-t0)*1000, 1))
            detector.confidence_threshold = 0.5
        break

# Test 3: List images in most recent scan
print("\n--- Test 3: Images in most recent scan ---")
if scan_dirs:
    all_imgs = sorted(glob.glob(os.path.join(scan_dirs[-1], "detected_images", "*.jpg")))
    print("  Total images:", len(all_imgs))
    for f in all_imgs[:10]:
        print("  ", os.path.basename(f))

print("\nDetect calls:", detector._detect_calls)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_detect_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    # Run using the venv activation, same as the service
    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_detect_test.py 2>&1", "Direct detection test", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
