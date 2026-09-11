#!/usr/bin/env python3
"""Verify v3 model is now loading in the service."""
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
    print("Connected. Waiting 45s for service to load model...")
    time.sleep(45)

    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 25 --no-pager 2>&1",
        timeout=30
    )
    print("\nLogs:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
