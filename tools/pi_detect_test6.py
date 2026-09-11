#!/usr/bin/env python3
"""SSH into the Pi and run detection test from the correct working directory."""
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

    # Write a test script that runs from ~/anilag like main.py does
    test_script = r'''import os, sys, time, glob
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, os.path.dirname(os.path.abspath("main.py")))

# Load env vars
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

# Find stored scan images
scan_dirs = sorted(glob.glob("previous_scans/scan_*"))
print("\nFound", len(scan_dirs), "scan directories")

# Test on stored images from the most recent scan
if scan_dirs:
    for d in scan_dirs[-2:]:
        imgs = sorted(glob.glob(os.path.join(d, "detected_images", "*.jpg")))
        # Get left and right annotated images (not preprocessed)
        left_imgs = [f for f in imgs if "_left_" in f and "preprocessed" not in f]
        right_imgs = [f for f in imgs if "_right_" in f and "preprocessed" not in f]

        for label, img_list in [("LEFT", left_imgs), ("RIGHT", right_imgs)]:
            if img_list:
                test_img = img_list[-1]
                print("\n--- " + label + " camera: " + os.path.basename(test_img) + " ---")
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
                            for i, (box, conf) in enumerate(zip(result.boxes[:5], result.confidences[:5])):
                                x1, y1, x2, y2 = box
                                w = x2 - x1
                                h = y2 - y1
                                area = w * h
                                frame_area = img.shape[0] * img.shape[1]
                                pct = area / frame_area * 100
                                print("    Box " + str(i) + ": " + str(box) + " " + str(w) + "x" + str(h) + " = " + str(area) + "px (" + str(round(pct, 2)) + "% of frame) conf=" + str(round(conf, 3)))
                        print("  Inference ms:", round((t1-t0)*1000, 1))
                    detector.confidence_threshold = 0.5
        break  # only test most recent scan

# Also test with max_box_area_ratio = 0.25 (tighter filter)
print("\n--- Test with max_box_area_ratio=0.25 (tighter) ---")
detector._max_box_area_ratio = 0.25
if scan_dirs:
    for d in scan_dirs[-1:]:
        imgs = sorted(glob.glob(os.path.join(d, "detected_images", "*.jpg")))
        right_imgs = [f for f in imgs if "_right_" in f and "preprocessed" not in f]
        if right_imgs:
            test_img = right_imgs[-1]
            print("  Testing:", os.path.basename(test_img))
            img = cv2.imread(test_img)
            if img is not None:
                detector.confidence_threshold = 0.25
                result = detector.detect(img)
                print("  Count:", result.count)
                if result.boxes:
                    for i, (box, conf) in enumerate(zip(result.boxes[:5], result.confidences[:5])):
                        x1, y1, x2, y2 = box
                        w = x2 - x1
                        h = y2 - y1
                        area = w * h
                        frame_area = img.shape[0] * img.shape[1]
                        pct = area / frame_area * 100
                        print("    Box " + str(i) + ": " + str(w) + "x" + str(h) + " = " + str(area) + "px (" + str(round(pct, 2)) + "% of frame) conf=" + str(round(conf, 3)))
                detector.confidence_threshold = 0.5

print("\nDetect calls:", detector._detect_calls)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_detect_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_detect_test.py 2>&1", "Direct detection test", timeout=280)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
