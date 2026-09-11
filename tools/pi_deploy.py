#!/usr/bin/env python3
"""Deploy updated files to the Pi and restart the service."""
import paramiko
import os
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

LOCAL_DIR = r"C:\Users\Eloisa\CascadeProjects\anilag"
REMOTE_DIR = "/home/user/anilag"

def run(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return out, err

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Verify remote directory exists
    out, err = run(client, f"ls -d {REMOTE_DIR}/src/detection {REMOTE_DIR}/src/gui 2>&1")
    print(f"Remote dirs check: {out}")

    # Upload files via SFTP using absolute paths
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
    print()

    # Restart the service
    print("Restarting anilag.service...")
    out, err = run(client, "sudo systemctl restart anilag.service && sleep 2 && sudo systemctl is-active anilag.service", timeout=30)
    print(f"Status: {out}")
    if err:
        print(f"stderr: {err}")

    # Wait for model to load and check logs
    print("\nWaiting 45s for model to load...")
    time.sleep(45)

    out, err = run(client, "sudo journalctl -u anilag.service -n 30 --no-pager 2>&1", timeout=30)
    print("\n=== Recent logs ===")
    print(out)

    # Check the deployed config
    out, err = run(client, f"grep -E 'CONFIDENCE_THRESHOLD|YOLO_MIN_BOX_AREA|YOLO_MAX_BOX_AREA_RATIO' {REMOTE_DIR}/config.env")
    print("\n=== Deployed config ===")
    print(out)

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
