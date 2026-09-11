#!/usr/bin/env python3
"""Test v3 model loading with Qt offscreen (like the service)."""
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

    # Test script that runs the full app with offscreen Qt
    test_script = r'''import os, sys, time, signal, faulthandler
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def timeout_handler(signum, frame):
    print("TIMEOUT: App took too long!")
    import traceback
    traceback.print_stack()
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

from dotenv import load_dotenv
load_dotenv("config.env")

print("Starting app with Qt offscreen...")
print(f"MODEL_PATH: {os.getenv('MODEL_PATH')}")

import main as app_main
print("main module loaded, calling main()...")
try:
    app_main.main()
except SystemExit:
    pass
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Exception: {e}")

signal.alarm(0)
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_v3_qt_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 90 python /tmp/pi_v3_qt_test.py 2>&1",
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
