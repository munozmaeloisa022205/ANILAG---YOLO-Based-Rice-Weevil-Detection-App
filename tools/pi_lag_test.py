#!/usr/bin/env python3
"""Measure actual detection cycle time on the Pi."""
import paramiko
import time
import sys

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Stop service to free cameras
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    test_script = r'''import os, sys, time
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
import numpy as np
from src.detection.yolov11_detector import YOLOv11Detector
from src.hardware.camera import DualCameraManager

detector = YOLOv11Detector(
    model_path="models/sitophilus_oryzae_v3_best.pt",
    confidence_threshold=0.08
)
if not detector.initialize():
    print("FAILED to initialize")
    sys.exit(1)

cam = DualCameraManager()
cam.start()
time.sleep(3)

print("\n--- Measuring detection cycle time (5 cycles) ---")
times = []
for i in range(5):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    t0 = time.perf_counter()
    if left is not None:
        result_l = detector.detect(left)
    if right is not None:
        result_r = detector.detect(right)
    t1 = time.perf_counter()
    cycle_ms = (t1 - t0) * 1000
    times.append(cycle_ms)
    total_count = max(result_l.count if left is not None else 0, result_r.count if right is not None else 0)
    print(f"Cycle {i+1}: {cycle_ms:.0f}ms  L={result_l.count if left is not None else 0}  R={result_r.count if right is not None else 0}  total={total_count}")

avg = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
print(f"\nAvg cycle (excl warmup): {avg:.0f}ms")
print(f"Detection FPS: {1000/avg:.1f}")

cam.stop()
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_lag_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 120 python /tmp/pi_lag_test.py 2>&1",
        timeout=150
    )
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    if err:
        sys.stdout.buffer.write(b"\nSTDERR:\n")
        sys.stdout.buffer.write(err.encode("utf-8", errors="replace"))

    # Restart service
    stdin, stdout, stderr = client.exec_command("sudo systemctl start anilag.service 2>&1")
    stdout.read()

    client.close()

if __name__ == "__main__":
    main()
