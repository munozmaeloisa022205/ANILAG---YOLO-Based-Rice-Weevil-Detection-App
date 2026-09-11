#!/usr/bin/env python3
"""Quick tiling comparison test - runs while service is active (uses service's detector)."""
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

    # Read recent detection logs from the service to measure cycle time
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '5 min ago' --no-pager --output=cat 2>&1 | grep -E '\\[DETECT\\]|inference_ms|cycle' | tail -20",
        timeout=30
    )
    out = stdout.read().decode(errors="replace")
    print("Recent DETECT logs from service:")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    # Also get all recent logs to see detection activity
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '5 min ago' --no-pager --output=cat 2>&1 | tail -30",
        timeout=30
    )
    out = stdout.read().decode(errors="replace")
    print("\nAll recent logs:")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
