#!/usr/bin/env python3
"""Wait longer and verify v3 model loaded with new changes."""
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
    print("Connected. Waiting 120s for model to load...")
    time.sleep(120)

    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 30 --no-pager --output=cat 2>&1",
        timeout=30
    )
    print("\nLogs:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
