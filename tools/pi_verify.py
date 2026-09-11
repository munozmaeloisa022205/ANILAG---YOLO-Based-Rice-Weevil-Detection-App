#!/usr/bin/env python3
"""Verify detection works with new config on the Pi."""
import paramiko
import time

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
print("Confidence:", detector.confidence_threshold)
print("Min box area:", detector._min_box_area)
print("Max box area ratio:", detector._max_box_area_ratio)
print("Max aspect ratio:", detector._max_aspect_ratio)

# Find ALL stored images
all_imgs = sorted(glob.glob("previous_scans/scan_*/detected_images/*.jpg"))
left_imgs = [f for f in all_imgs if "_left_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
right_imgs = [f for f in all_imgs if "_right_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
print("\nLeft images:", len(left_imgs), "Right images:", len(right_imgs))

# Test on most recent images from each camera
for label, img_list in [("LEFT", left_imgs), ("RIGHT", right_imgs)]:
    if img_list:
        for test_img in img_list[-2:]:
            print("\n--- " + label + ": " + os.path.basename(test_img) + " ---")
            img = cv2.imread(test_img)
            if img is None:
                print("  Could not read image")
                continue
            print("  Shape:", img.shape)

            t0 = time.perf_counter()
            result = detector.detect(img)
            t1 = time.perf_counter()
            print("  count=" + str(result.count) + " (" + str(round((t1-t0)*1000, 1)) + "ms)")
            if result.boxes:
                for i, (box, conf) in enumerate(zip(result.boxes[:8], result.confidences[:8])):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    frame_area = img.shape[0] * img.shape[1]
                    pct = area / frame_area * 100
                    print("    Box " + str(i) + ": " + str(w) + "x" + str(h) + " = " + str(area) + "px (" + str(round(pct, 2)) + "%) conf=" + str(round(conf, 3)))

print("\nDetect calls:", detector._detect_calls)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_verify_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_verify_test.py 2>&1", "Detection test with new config", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
