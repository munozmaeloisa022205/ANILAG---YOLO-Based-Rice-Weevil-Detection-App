#!/usr/bin/env python3
"""Test NCNN model loading directly."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # First stop the service to free resources
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    import time
    time.sleep(3)

    test_script = r'''import os, sys
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
print("Testing NCNN model loading...")
from src.detection.yolov11_detector import YOLOv11Detector
detector = YOLOv11Detector(confidence_threshold=0.08)
ok = detector.initialize()
print("Initialized:", ok)
if ok:
    import numpy as np
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(dummy)
    print("Detection test: count=", result.count)
else:
    print("FAILED to initialize")
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_model_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && python /tmp/pi_model_test.py 2>&1",
        timeout=60
    )
    print(stdout.read().decode(errors="replace"))
    err = stderr.read().decode(errors="replace")
    if err:
        print("STDERR:", err)

    # Restart the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl start anilag.service 2>&1")
    stdout.read()

    client.close()

if __name__ == "__main__":
    main()
