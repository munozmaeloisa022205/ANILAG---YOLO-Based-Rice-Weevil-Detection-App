#!/usr/bin/env python3
"""Verify detection with correct config loading on the Pi."""
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

    # Use dotenv to load config properly, like the app does
    test_script = r'''import os, sys, time, glob
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")

# Load config.env the same way the app does
from dotenv import load_dotenv
load_dotenv("config.env")

import numpy as np
import cv2
from src.detection.yolov11_detector import YOLOv11Detector

# Read config the same way the GUI does
conf = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
iou = float(os.getenv("IOU_THRESHOLD", "0.7"))
print("Env CONFIDENCE_THRESHOLD:", os.getenv("CONFIDENCE_THRESHOLD"))
print("Env YOLO_MIN_BOX_AREA:", os.getenv("YOLO_MIN_BOX_AREA"))
print("Env YOLO_MAX_BOX_AREA_RATIO:", os.getenv("YOLO_MAX_BOX_AREA_RATIO"))
print("Using confidence:", conf, "iou:", iou)

detector = YOLOv11Detector(
    confidence_threshold=conf,
    iou_threshold=iou,
)
if not detector.initialize():
    print("FAILED to initialize")
    sys.exit(1)

print("Detector confidence:", detector.confidence_threshold)
print("Detector min_box_area:", detector._min_box_area)
print("Detector max_box_area_ratio:", detector._max_box_area_ratio)

# Find stored images
all_imgs = sorted(glob.glob("previous_scans/scan_*/detected_images/*.jpg"))
left_imgs = [f for f in all_imgs if "_left_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
right_imgs = [f for f in all_imgs if "_right_" in os.path.basename(f) and "preprocessed" not in os.path.basename(f)]
print("\nLeft images:", len(left_imgs), "Right images:", len(right_imgs))

for label, img_list in [("LEFT", left_imgs), ("RIGHT", right_imgs)]:
    if img_list:
        for test_img in img_list[-2:]:
            print("\n--- " + label + ": " + os.path.basename(test_img) + " ---")
            img = cv2.imread(test_img)
            if img is None:
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
    with sftp.file("/tmp/pi_verify2.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_verify2.py 2>&1", "Detection test with dotenv", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
