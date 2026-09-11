#!/usr/bin/env python3
"""Test v3 model with tiling off to measure speed and recall."""
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

    # Stop service
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    test_script = r'''import os, sys, time
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
import numpy as np

# Override tiling to test both modes
from src.detection.yolov11_detector import YOLOv11Detector, _parse_tile_grid
from src.hardware.camera import DualCameraManager

cam = DualCameraManager()
cam.start()
time.sleep(3)

# Test 1: 2x2 tiling (current)
print("=== Test 1: 2x2 tiling (current) ===")
os.environ["YOLO_TILES"] = "2x2"
det = YOLOv11Detector(model_path="models/sitophilus_oryzae_v3_best.pt", confidence_threshold=0.08)
det.initialize()
print(f"Tiles: {det._tile_grid}")

for i in range(3):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    t0 = time.perf_counter()
    rl = det.detect(left) if left is not None else None
    rr = det.detect(right) if right is not None else None
    t1 = time.perf_counter()
    cycle_ms = (t1 - t0) * 1000
    lc = rl.count if rl else 0
    rc = rr.count if rr else 0
    print(f"  Cycle {i+1}: {cycle_ms:.0f}ms  L={lc}  R={rc}  max={max(lc,rc)}")
time.sleep(1)

# Test 2: Tiling off
print("\n=== Test 2: Tiling OFF ===")
os.environ["YOLO_TILES"] = "off"
det2 = YOLOv11Detector(model_path="models/sitophilus_oryzae_v3_best.pt", confidence_threshold=0.08)
det2.initialize()
print(f"Tiles: {det2._tile_grid}")

for i in range(3):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    t0 = time.perf_counter()
    rl = det2.detect(left) if left is not None else None
    rr = det2.detect(right) if right is not None else None
    t1 = time.perf_counter()
    cycle_ms = (t1 - t0) * 1000
    lc = rl.count if rl else 0
    rc = rr.count if rr else 0
    print(f"  Cycle {i+1}: {cycle_ms:.0f}ms  L={lc}  R={rc}  max={max(lc,rc)}")
time.sleep(1)

# Test 3: 1x2 tiling (horizontal split only)
print("\n=== Test 3: 1x2 tiling (horizontal) ===")
os.environ["YOLO_TILES"] = "1x2"
det3 = YOLOv11Detector(model_path="models/sitophilus_oryzae_v3_best.pt", confidence_threshold=0.08)
det3.initialize()
print(f"Tiles: {det3._tile_grid}")

for i in range(3):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    t0 = time.perf_counter()
    rl = det3.detect(left) if left is not None else None
    rr = det3.detect(right) if right is not None else None
    t1 = time.perf_counter()
    cycle_ms = (t1 - t0) * 1000
    lc = rl.count if rl else 0
    rc = rr.count if rr else 0
    print(f"  Cycle {i+1}: {cycle_ms:.0f}ms  L={lc}  R={rc}  max={max(lc,rc)}")

cam.stop()
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_tiling_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 180 python /tmp/pi_tiling_test.py 2>&1",
        timeout=210
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
