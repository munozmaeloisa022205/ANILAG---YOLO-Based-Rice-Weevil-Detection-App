#!/usr/bin/env python3
"""Check Pi state and kill stuck processes."""
import paramiko
import time
import sys

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return out, err

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Kill any stuck test processes
    run(client, "pkill -9 -f 'pi_tiling_test' 2>/dev/null; pkill -9 -f 'pi_lag_test' 2>/dev/null; echo killed")

    # Check service status
    out, _ = run(client, "sudo systemctl is-active anilag.service 2>&1")
    print("Service status:", out.strip())

    # Check running processes
    out, _ = run(client, "ps aux | grep python | grep -v grep 2>&1")
    print("\nPython processes:")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    # Get recent logs
    out, _ = run(client, "sudo journalctl -u anilag.service -n 15 --no-pager --output=cat 2>&1")
    print("\nRecent logs:")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
