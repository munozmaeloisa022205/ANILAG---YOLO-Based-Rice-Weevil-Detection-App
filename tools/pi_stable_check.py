#!/usr/bin/env python3
"""Verify service is stable."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected. Waiting 20s...")
    time.sleep(20)

    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '2 min ago' --no-pager 2>&1 | tail -15",
        timeout=30
    )
    print("\nLogs:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
