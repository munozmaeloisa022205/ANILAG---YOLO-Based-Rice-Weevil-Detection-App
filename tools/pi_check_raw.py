#!/usr/bin/env python3
"""Check that raw images are stored in the database."""
import paramiko
import time

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

    # Check the database for image types in recent scans
    test_script = r'''import os, sys, glob
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv("config.env")
from backend.database import DatabaseManager
db = DatabaseManager()

# Get the most recent scans
scans = db.get_all_scans()
print("Total scans:", len(scans))
if scans:
    scan_id = scans[-1]["scan_id"]
    print("\nMost recent scan:", scan_id)
    images = db.get_scan_images(scan_id, include_blob=False)
    print("Total images:", len(images))
    # Group by camera type
    by_type = {}
    for img in images:
        cam = img.get("camera", "?")
        by_type.setdefault(cam, []).append(img.get("filename", "?"))
    for cam, files in sorted(by_type.items()):
        print(f"  {cam}: {len(files)} images")
        for f in files[:3]:
            print(f"    {f}")
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_check_raw.py", "w") as f:
        f.write(test_script)
    sftp.close()

    run(client, "cd ~/anilag && source venv/bin/activate && python /tmp/pi_check_raw.py 2>&1", "Database image types", timeout=60)

    # Also check the most recent scan folder on disk
    run(client, "ls ~/anilag/previous_scans/scan_*/detected_images/*_raw_* 2>/dev/null | tail -20", "Raw images on disk")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
