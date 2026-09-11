#!/usr/bin/env python3
"""Get full crash error."""
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

    # Get full crash logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '3 min ago' --no-pager 2>&1 | tail -80",
        timeout=30
    )
    print(stdout.read().decode(errors="replace"))

    # Try running the app directly to see the error
    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 30 python main.py 2>&1 | tail -40",
        timeout=45
    )
    print("=== Direct run ===")
    print(stdout.read().decode(errors="replace"))
    print(stderr.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
