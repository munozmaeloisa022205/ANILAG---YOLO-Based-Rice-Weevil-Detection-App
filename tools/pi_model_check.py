#!/usr/bin/env python3
"""Check NCNN model files and clear leftover state."""
import paramiko
import time

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

    # Kill any leftover Python processes
    stdin, stdout, stderr = client.exec_command("pkill -9 -f 'python.*main.py' 2>&1; sleep 2; echo done")
    print("Kill:", stdout.read().decode().strip())

    # Check NCNN model files
    stdin, stdout, stderr = client.exec_command(
        "ls -la ~/anilag/models/sitophilus_oryzae_v2-3_best_ncnn_model/ 2>&1"
    )
    print("Model files:")
    print(stdout.read().decode(errors="replace"))

    # Check if there are any lock files
    stdin, stdout, stderr = client.exec_command(
        "find ~/anilag/models -name '*.lock' -o -name '*.tmp' 2>/dev/null"
    )
    locks = stdout.read().decode().strip()
    print("Lock files:", locks or "none")

    # Check the detector file on Pi matches local
    stdin, stdout, stderr = client.exec_command(
        "head -5 ~/anilag/src/detection/yolov11_detector.py 2>&1"
    )
    print("Detector header:")
    print(stdout.read().decode(errors="replace"))

    # Try running the full app again with faulthandler
    test_script = r'''import os, sys, faulthandler
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from dotenv import load_dotenv
load_dotenv("config.env")
print("Importing detector...")
from src.detection.yolov11_detector import YOLOv11Detector
print("Creating detector...")
detector = YOLOv11Detector(confidence_threshold=0.08)
print("Initializing...")
ok = detector.initialize()
print("Initialized:", ok)
if ok:
    import numpy as np
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    result = detector.detect(dummy)
    print("Detection test: count=", result.count)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_model_test2.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 30 python /tmp/pi_model_test2.py 2>&1",
        timeout=45
    )
    print("\n=== Model test with src import ===")
    print(stdout.read().decode(errors="replace"))
    err = stderr.read().decode(errors="replace")
    if err:
        print("STDERR:", err)

    client.close()

if __name__ == "__main__":
    main()
