#!/usr/bin/env python3
"""Quick test: v3 model with tiling OFF vs 2x2. Stops service, tests, restarts."""
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
os.environ["YOLO_TILES"] = "off"

from src.detection.yolov11_detector import YOLOv11Detector
from src.hardware.camera import DualCameraManager

det = YOLOv11Detector(model_path="models/sitophilus_oryzae_v3_best.pt", confidence_threshold=0.08)
det.initialize()
print(f"Tiles: {det._tile_grid}")

cam = DualCameraManager()
cam.start()
time.sleep(3)

# Test tiling OFF
print("\n=== Tiling OFF (1 pass per camera) ===")
for i in range(3):
    left = cam.get_left_frame()
    right = cam.get_right_frame()
    t0 = time.perf_counter()
    rl = det.detect(left) if left is not None else None
    rr = det.detect(right) if right is not None else None
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000
    lc = rl.count if rl else 0
    rc = rr.count if rr else 0
    print(f"  Cycle {i+1}: {ms:.0f}ms  L={lc}  R={rc}  max={max(lc,rc)}")

# Now test 2x2 for comparison
print("\n=== 2x2 tiling (5 passes per camera) ===")
os.environ["YOLO_TILES"] = "2x2"
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
    ms = (t1 - t0) * 1000
    lc = rl.count if rl else 0
    rc = rr.count if rr else 0
    print(f"  Cycle {i+1}: {ms:.0f}ms  L={lc}  R={rc}  max={max(lc,rc)}")

cam.stop()
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_quick_tiling.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 120 python /tmp/pi_quick_tiling.py 2>&1",
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
    print("\nService restarted.")

    client.close()

if __name__ == "__main__":
    main()
