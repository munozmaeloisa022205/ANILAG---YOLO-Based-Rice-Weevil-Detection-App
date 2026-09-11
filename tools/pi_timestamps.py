#!/usr/bin/env python3
"""Check timestamps of stored images vs service restart."""
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

    # Check image timestamps
    run(client, "ls -la --time-style=full-iso ~/anilag/previous_scans/scan_2026-09-08_17-15-24/detected_images/ 2>/dev/null", "Image timestamps")

    # Check service start time
    run(client, "sudo systemctl show anilag.service -p ActiveEnterTimestamp 2>&1", "Service start time")

    # Check for scans that happened after 17:15:24
    run(client, "find ~/anilag/previous_scans -maxdepth 1 -type d -newer ~/anilag/config.env | sort", "Scans after config update")

    # Check the most recent scan directory
    run(client, "ls -dt ~/anilag/previous_scans/scan_* | head -3", "Most recent scan dirs")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
