#!/usr/bin/env python3
"""SSH into the Pi and check how the service runs, then test detection."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=240):
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

    # Check the service file to see how it runs Python
    run(client, "cat ~/anilag/deploy/anilag.service 2>/dev/null; cat /etc/systemd/system/anilag.service 2>/dev/null", "Service file")

    # Check for venv or pip packages
    run(client, "ls ~/anilag/.venv 2>/dev/null; pip3 list 2>/dev/null | grep -iE 'opencv|ultralytics|ncnn|torch|numpy' | head -10; python3 -c 'import cv2; print(cv2.__version__)' 2>&1", "Python packages")

    # Check if there's a requirements.txt
    run(client, "cat ~/anilag/requirements.txt 2>/dev/null | head -20", "Requirements")

    # Check how the service actually runs
    run(client, "systemctl cat anilag.service 2>/dev/null", "Systemctl cat")

    # Find any python with cv2
    run(client, "find / -name 'python3*' -type f 2>/dev/null | head -10; find / -path '*/site-packages/cv2*' -maxdepth 8 2>/dev/null | head -5", "Find Python with cv2")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
