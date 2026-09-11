#!/usr/bin/env python3
"""Check if v3 model is still loading or stuck."""
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

    # Check if the process is running
    stdin, stdout, stderr = client.exec_command("ps aux | grep -E 'python.*main' | grep -v grep 2>&1")
    print("Process:")
    print(stdout.read().decode(errors="replace"))

    # Check service status with more detail
    stdin, stdout, stderr = client.exec_command("sudo systemctl status anilag.service 2>&1 | head -15")
    print("\nService status:")
    print(stdout.read().decode(errors="replace"))

    # Get all logs since the last restart
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '01:00' --no-pager 2>&1",
        timeout=30
    )
    print("\nAll logs since restart:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
