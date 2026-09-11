#!/usr/bin/env python3
"""Verify the new config is loaded after restart."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=30):
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

    # Wait for model to load
    print("Waiting 30s for model to load...")
    time.sleep(30)

    # Check the latest logs for the new config
    run(client, "sudo journalctl -u anilag.service --since '2 min ago' --no-pager 2>&1 | grep -E 'Tiled|Confidence|Box filters' | tail -5", "New config in logs")

    # Check config file on Pi
    run(client, "grep 'YOLO_TILES' ~/anilag/config.env", "Config YOLO_TILES")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
