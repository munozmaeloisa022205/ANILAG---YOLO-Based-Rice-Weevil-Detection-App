#!/usr/bin/env python3
"""Check for errors after deploy."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected. Waiting 30s for model load...")
    time.sleep(30)

    stdin, stdout, stderr = client.exec_command(
        'sudo journalctl -u anilag.service --since "2 min ago" --no-pager 2>&1 | grep -iE "error|traceback|valueerror|exception" | tail -10',
        timeout=30
    )
    out = stdout.read().decode(errors="replace")
    print("Errors:", out.strip() or "None")

    stdin, stdout, stderr = client.exec_command(
        'sudo journalctl -u anilag.service --since "2 min ago" --no-pager 2>&1 | grep -E "Confidence|Box filters|Tiled" | tail -3',
        timeout=30
    )
    print("\nConfig:", stdout.read().decode(errors="replace").strip())

    client.close()

if __name__ == "__main__":
    main()
