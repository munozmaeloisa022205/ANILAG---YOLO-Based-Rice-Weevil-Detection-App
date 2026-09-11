#!/usr/bin/env python3
"""Check live service logs for detection output."""
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

    # Check for DETECT/INFER/ERROR log lines since the last restart
    run(client, "sudo journalctl -u anilag.service --since '5 min ago' --no-pager 2>&1 | grep -E 'DETECT|INFER|ERROR|Count|detect' | tail -30", "Detection logs (last 5 min)")

    # Check all recent logs
    run(client, "sudo journalctl -u anilag.service --since '5 min ago' --no-pager 2>&1 | tail -40", "All recent logs")

    # Check if any scans were run recently
    run(client, "find ~/anilag/previous_scans -name '*.jpg' -newer ~/anilag/config.env 2>/dev/null | head -20", "Recent stored images")

    # Check the database for recent detections
    run(client, "cd ~/anilag && source venv/bin/activate && python -c \"import os; os.chdir('/home/user/anilag'); from dotenv import load_dotenv; load_dotenv('config.env'); import sys; sys.path.insert(0,'.'); from backend.database import DatabaseManager; db=DatabaseManager(); scans=db.get_all_scans(); print('Total scans:', len(scans)); [print('  ', s['scan_id'], 'max_count=', s.get('max_count','?'), 'images=', s.get('image_count','?')) for s in scans[-5:]]\" 2>&1", "Database scan summary")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
