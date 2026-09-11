#!/usr/bin/env python3
"""Wait for v3 model to finish loading."""
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
    print("Connected. Waiting 120s for model to finish loading...")
    time.sleep(120)

    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '01:00' --no-pager 2>&1",
        timeout=30
    )
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")

    client.close()

if __name__ == "__main__":
    main()
