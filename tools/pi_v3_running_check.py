#!/usr/bin/env python3
"""Check if v3 service is actually running and producing scans."""
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
    print("Connected.\n")

    # Check process
    stdin, stdout, stderr = client.exec_command("ps aux | grep 'python.*main' | grep -v grep 2>&1")
    print("Process:")
    sys.stdout.buffer.write(stdout.read().decode(errors="replace").encode("utf-8", errors="replace"))

    # Check for new scan directories
    stdin, stdout, stderr = client.exec_command("ls -dt ~/anilag/previous_scans/scan_* 2>/dev/null | head -3")
    print("\nRecent scans:")
    sys.stdout.buffer.write(stdout.read().decode(errors="replace").encode("utf-8", errors="replace"))

    # Check full logs with no time filter
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 40 --no-pager --output=cat 2>&1",
        timeout=30
    )
    print("\nFull logs (raw):")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
