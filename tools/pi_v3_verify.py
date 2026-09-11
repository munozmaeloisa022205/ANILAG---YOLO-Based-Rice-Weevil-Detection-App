#!/usr/bin/env python3
"""Verify the v3 model loaded successfully on the Pi."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected. Waiting 20s for model to finish loading...")
    time.sleep(20)

    # Check service status
    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    # Check logs for model loaded confirmation
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '2 min ago' --no-pager 2>&1 | grep -E 'model loaded|Confidence|Box filters|Tiled|error|Error|crash|Crash' | tail -10",
        timeout=30
    )
    print("\nModel load logs:")
    print(stdout.read().decode(errors="replace"))

    # Check for any errors
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '2 min ago' --no-pager 2>&1 | grep -iE 'error|traceback|exception|crash|aborted' | tail -5",
        timeout=30
    )
    errors = stdout.read().decode(errors="replace").strip()
    print("Errors:", errors or "None")

    client.close()

if __name__ == "__main__":
    main()
