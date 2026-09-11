#!/usr/bin/env python3
"""Check v3 model logs with proper encoding."""
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

    # Get all logs since the last restart
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '01:00' --no-pager 2>&1",
        timeout=30
    )
    out = stdout.read().decode(errors="replace")
    # Write to stdout with utf-8
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")

    client.close()

if __name__ == "__main__":
    main()
