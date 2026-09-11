#!/usr/bin/env python3
"""Check all recent logs to see detection activity."""
import paramiko

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None, timeout=30):
    if label:
        print(f"\n=== {label} ===")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if out.strip():
        print(out.rstrip())
    if err.strip():
        print(f"[stderr] {err.rstrip()}")
    return out

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Get ALL logs since the last restart - look for detection output
    run(client, "sudo journalctl -u anilag.service --since '20 min ago' --no-pager 2>&1 | grep -v 'cap_v4l\\|cap.cpp\\|XDG_RUNTIME\\|systemd\\[1\\]' | tail -60", "All detection-related logs")

    # Check if the scan is still running or finished
    run(client, "sudo journalctl -u anilag.service --since '20 min ago' --no-pager 2>&1 | grep -i 'scan\\|stop\\|start\\|email\\|report' | tail -20", "Scan lifecycle logs")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
