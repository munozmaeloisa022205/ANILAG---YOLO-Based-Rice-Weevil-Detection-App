#!/usr/bin/env python3
"""Wait for model to load and verify the new changes."""
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
    print("Connected. Waiting 10 min for model to load (v3 is slow on first service start)...")

    # Poll every 60s for up to 10 minutes
    for i in range(10):
        time.sleep(60)
        stdin, stdout, stderr = client.exec_command(
            "sudo journalctl -u anilag.service -n 5 --no-pager --output=cat 2>&1 | grep -E 'model loaded|Tiled|error|Error'",
            timeout=10
        )
        out = stdout.read().decode(errors="replace").strip()
        if "model loaded" in out:
            print(f"\nMinute {i+1}: Model loaded!")
            print(out)
            break
        elif out:
            print(f"Minute {i+1}: {out}")
        else:
            print(f"Minute {i+1}: still loading...")

    # Get full logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 30 --no-pager --output=cat 2>&1",
        timeout=30
    )
    print("\nFull logs:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
