#!/usr/bin/env python3
"""Check process state and get all logs."""
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

    # Check process
    stdin, stdout, stderr = client.exec_command("ps aux | grep 'python.*main' | grep -v grep 2>&1")
    print("Process:")
    sys.stdout.buffer.write(stdout.read().decode(errors="replace").encode("utf-8", errors="replace"))

    # Get ALL logs (no time filter, raw output)
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 50 --no-pager --output=cat 2>&1",
        timeout=30
    )
    print("\nAll logs:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
