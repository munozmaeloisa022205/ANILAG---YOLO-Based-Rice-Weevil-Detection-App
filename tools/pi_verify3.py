#!/usr/bin/env python3
"""Wait for model load and verify config, then run detection test."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=30):
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

    # Wait longer for model to load (NCNN load takes ~30s on Pi)
    print("Waiting 40s for model to fully load...")
    time.sleep(40)

    # Check the latest logs
    run(client, "sudo journalctl -u anilag.service --since '3 min ago' --no-pager 2>&1 | grep -E 'Tiled|Confidence|Box filters|model loaded' | tail -5", "Config in logs")

    # Run a quick detection test with the new 2x2 config
    test_script = r'''import os, sys, time, glob
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
import numpy as np
import cv2
from src.detection.yolov11_detector import YOLOv11Detector

conf = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
detector = YOLOv11Detector(confidence_threshold=conf, iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.7")))
if not detector.initialize():
    print("FAILED to initialize")
    sys.exit(1)

print("Tiles:", detector._tile_grid)
print("Confidence:", detector.confidence_threshold)
print("Max ratio:", detector._max_box_area_ratio)

# Test on a stored image
all_imgs = sorted(glob.glob("previous_scans/scan_*/detected_images/*_left_*.jpg"))
left_imgs = [f for f in all_imgs if "preprocessed" not in f and "raw" not in f]
if left_imgs:
    test_img = left_imgs[-1]
    print("\nTesting:", os.path.basename(test_img))
    img = cv2.imread(test_img)
    if img is not None:
        t0 = time.perf_counter()
        result = detector.detect(img)
        t1 = time.perf_counter()
        print("Count:", result.count)
        print("Inference ms:", round((t1-t0)*1000, 1))
        if result.boxes:
            for i, (box, c) in enumerate(zip(result.boxes[:5], result.confidences[:5])):
                x1, y1, x2, y2 = box
                print("  Box", i, ":", str(x2-x1) + "x" + str(y2-y1), "conf=" + str(round(c, 3)))
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_quick_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_quick_test.py 2>&1", "Quick detection test (2x2 tiling)", timeout=120)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
