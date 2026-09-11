#!/usr/bin/env python3
"""Test v3 NCNN model loading directly on Pi."""
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

    # Stop the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    # Test script with timeout
    test_script = r'''import os, sys, time, signal
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")

def timeout_handler(signum, frame):
    print("TIMEOUT: Model loading took too long!")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout

from dotenv import load_dotenv
load_dotenv("config.env")

print("Testing v3 NCNN model loading...")
print("Model path:", os.getenv("MODEL_PATH"))

from src.detection.yolov11_detector import YOLOv11Detector
detector = YOLOv11Detector(confidence_threshold=0.08)

print("Calling initialize()...")
t0 = time.time()
ok = detector.initialize()
t1 = time.time()
print(f"Initialized: {ok} in {t1-t0:.1f}s")

if ok:
    import numpy as np
    print("Running test detection...")
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    t0 = time.time()
    result = detector.detect(dummy)
    t1 = time.time()
    print(f"Detection: count={result.count}, time={t1-t0:.1f}s")
else:
    print("FAILED to initialize")

signal.alarm(0)  # cancel alarm
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_v3_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 90 python /tmp/pi_v3_test.py 2>&1",
        timeout=120
    )
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    if err:
        sys.stdout.buffer.write(b"\nSTDERR:\n")
        sys.stdout.buffer.write(err.encode("utf-8", errors="replace"))

    # Restart the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl start anilag.service 2>&1")
    stdout.read()

    client.close()

if __name__ == "__main__":
    main()
