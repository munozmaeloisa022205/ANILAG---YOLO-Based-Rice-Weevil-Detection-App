#!/usr/bin/env python3
"""Get full recent logs from Pi."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '3 min ago' --no-pager 2>&1 | tail -25",
        timeout=30
    )
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
