#!/usr/bin/env python3
"""Check stored images from the latest scan on the Pi."""
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

    # List all images in the most recent scan
    run(client, "ls -la ~/anilag/previous_scans/scan_2026-09-08_17-15-24/detected_images/ 2>/dev/null | head -30", "Latest scan images")

    # Count by camera
    run(client, "ls ~/anilag/previous_scans/scan_2026-09-08_17-15-24/detected_images/ 2>/dev/null | grep -c '_left_' | head -1; ls ~/anilag/previous_scans/scan_2026-09-08_17-15-24/detected_images/ 2>/dev/null | grep -c '_right_' | head -1", "Left vs Right image count")

    # Show right camera images specifically
    run(client, "ls ~/anilag/previous_scans/scan_2026-09-08_17-15-24/detected_images/ 2>/dev/null | grep '_right_' | head -20", "Right camera images")

    # Check DETECT diagnostic logs from the scan
    run(client, "sudo journalctl -u anilag.service --since '10 min ago' --no-pager 2>&1 | grep -E '\\[DETECT\\]|\\[INFER\\]|\\[DETECT ERROR\\]' | tail -20", "DETECT diagnostic logs")

    # Check all logs with Count
    run(client, "sudo journalctl -u anilag.service --since '10 min ago' --no-pager 2>&1 | grep -i 'count' | tail -20", "Count logs")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
