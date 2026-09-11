#!/usr/bin/env python3
"""Clear pycache and redeploy."""
import paramiko
import os
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"
LOCAL_DIR = r"C:\Users\Eloisa\CascadeProjects\anilag"
REMOTE_DIR = "/home/user/anilag"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Stop the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    # Clear ALL __pycache__ directories
    stdin, stdout, stderr = client.exec_command(
        f"find {REMOTE_DIR} -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; echo done"
    )
    print("Clear pycache:", stdout.read().decode().strip())

    # Upload files
    sftp = client.open_sftp()
    files = [
        ("config.env", f"{REMOTE_DIR}/config.env"),
        ("src/detection/yolov11_detector.py", f"{REMOTE_DIR}/src/detection/yolov11_detector.py"),
        ("src/gui/main_window.py", f"{REMOTE_DIR}/src/gui/main_window.py"),
    ]
    for local, remote in files:
        local_path = os.path.join(LOCAL_DIR, local)
        print(f"Uploading {local} -> {remote}")
        sftp.put(local_path, remote)
    sftp.close()

    # Start the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl start anilag.service 2>&1")
    stdout.read()
    time.sleep(40)

    # Check status
    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    status = stdout.read().decode().strip()
    print("Status:", status)

    # Check logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '1 min ago' --no-pager 2>&1 | tail -20",
        timeout=30
    )
    print("\nLogs:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
