#!/usr/bin/env python3
"""SSH into the Pi and run detection on stored images from ANY scan."""
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

    test_script = r'''import os, sys, time, glob
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")

from pathlib import Path
env_file = Path("config.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import numpy as np
import cv2
from src.detection.yolov11_detector import YOLOv11Detector

detector = YOLOv11Detector()
if not detector.initialize():
    print("FAILED to initialize")
    sys.exit(1)

print("NCNN:", detector.use_ncnn, "CLAHE:", detector.use_clahe)
print("Confidence:", detector.confidence_threshold, "Max ratio:", detector._max_box_area_ratio)

# Find ALL stored images across ALL scans
all_imgs = sorted(glob.glob("previous_scans/scan_*/detected_images/*.jpg"))
print("\nTotal stored images:", len(all_imgs))

# Find non-preprocessed left and right images
left_imgs = [f for f in all_imgs if "_left_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
right_imgs = [f for f in all_imgs if "_right_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
print("Left images:", len(left_imgs))
print("Right images:", len(right_imgs))

# Test on a few images from each camera
for label, img_list in [("LEFT", left_imgs), ("RIGHT", right_imgs)]:
    if img_list:
        # Test the most recent 2 images
        for test_img in img_list[-2:]:
            print("\n--- " + label + ": " + os.path.basename(test_img) + " ---")
            img = cv2.imread(test_img)
            if img is None:
                print("  Could not read image")
                continue
            print("  Shape:", img.shape)

            for conf_val in [0.5, 0.25, 0.1, 0.05]:
                detector.confidence_threshold = conf_val
                t0 = time.perf_counter()
                result = detector.detect(img)
                t1 = time.perf_counter()
                print("  conf=" + str(conf_val) + " -> count=" + str(result.count) + " (" + str(round((t1-t0)*1000, 1)) + "ms)")
                if result.boxes:
                    for i, (box, conf) in enumerate(zip(result.boxes[:5], result.confidences[:5])):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        frame_area = img.shape[0] * img.shape[1]
                        pct = area / frame_area * 100
                        print("    Box " + str(i) + ": " + str(w) + "x" + str(h) + " = " + str(area) + "px (" + str(round(pct, 1)) + "%) conf=" + str(round(conf, 3)))
            detector.confidence_threshold = 0.5

# Also test with tighter max_box_area_ratio
print("\n--- Test with max_box_area_ratio=0.25 ---")
detector._max_box_area_ratio = 0.25
if right_imgs:
    test_img = right_imgs[-1]
    print("  Image:", os.path.basename(test_img))
    img = cv2.imread(test_img)
    if img is not None:
        for conf_val in [0.25, 0.1]:
            detector.confidence_threshold = conf_val
            result = detector.detect(img)
            print("  conf=" + str(conf_val) + " -> count=" + str(result.count))
            if result.boxes:
                for i, (box, conf) in enumerate(zip(result.boxes[:5], result.confidences[:5])):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    frame_area = img.shape[0] * img.shape[1]
                    pct = area / frame_area * 100
                    print("    Box " + str(i) + ": " + str(w) + "x" + str(h) + " = " + str(area) + "px (" + str(round(pct, 1)) + "%) conf=" + str(round(conf, 3)))
    detector.confidence_threshold = 0.5
    detector._max_box_area_ratio = 0.85

print("\nDetect calls:", detector._detect_calls)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_detect_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_detect_test.py 2>&1", "Detection test on stored images", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
