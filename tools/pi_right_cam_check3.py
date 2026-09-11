#!/usr/bin/env python3
"""Check right camera detection - stop service first to free cameras."""
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

    # Stop the service to free the cameras
    run(client, "sudo systemctl stop anilag.service 2>&1", "Stopping service")
    time.sleep(3)

    test_script = r'''import os, sys, time
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
import numpy as np
import cv2
from src.detection.yolov11_detector import YOLOv11Detector
from src.hardware.camera import DualCameraManager

detector = YOLOv11Detector(confidence_threshold=0.08)
if not detector.initialize():
    print("FAILED to initialize detector")
    sys.exit(1)

print("Tiles:", detector._tile_grid)
print("Confidence:", detector.confidence_threshold)

cam = DualCameraManager()
cam.start()
time.sleep(3)

print("\n--- Capturing 3 frames from each camera ---")
for i in range(3):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    print(f"\nFrame {i+1}:")
    if left is not None:
        t0 = time.perf_counter()
        result_l = detector.detect(left)
        t1 = time.perf_counter()
        print(f"  Left: count={result_l.count}, time={round((t1-t0)*1000,1)}ms")
        for j, (box, c) in enumerate(zip(result_l.boxes[:5], result_l.confidences[:5])):
            x1, y1, x2, y2 = box
            print(f"    Box {j}: {int(x2-x1)}x{int(y2-y1)}px conf={round(c,3)}")
    else:
        print("  Left: no frame")
    if right is not None:
        t0 = time.perf_counter()
        result_r = detector.detect(right)
        t1 = time.perf_counter()
        print(f"  Right: count={result_r.count}, time={round((t1-t0)*1000,1)}ms")
        for j, (box, c) in enumerate(zip(result_r.boxes[:5], result_r.confidences[:5])):
            x1, y1, x2, y2 = box
            print(f"    Box {j}: {int(x2-x1)}x{int(y2-y1)}px conf={round(c,3)}")
    else:
        print("  Right: no frame")
    time.sleep(1)

cam.stop()
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_right_cam_test3.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && timeout 90 python /tmp/pi_right_cam_test3.py 2>&1", "Live right camera detection test", timeout=120)

    # Restart the service
    run(client, "sudo systemctl start anilag.service 2>&1", "Restarting service")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
