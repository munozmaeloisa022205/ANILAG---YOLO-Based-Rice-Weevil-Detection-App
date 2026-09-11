#!/usr/bin/env python3
"""Check if v3 model is still loading or crashed."""
import paramiko
import sys

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Check if process is still running
    stdin, stdout, stderr = client.exec_command("ps aux | grep 'python.*main' | grep -v grep 2>&1")
    print("Process:")
    sys.stdout.buffer.write(stdout.read().decode(errors="replace").encode("utf-8", errors="replace"))

    # Get ALL recent logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 30 --no-pager 2>&1",
        timeout=30
    )
    print("\nLogs:")
    sys.stdout.buffer.write(stdout.read().decode(errors="replace").encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
