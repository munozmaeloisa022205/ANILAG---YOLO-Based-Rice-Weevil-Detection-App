#!/usr/bin/env python3
"""Run the full app with offscreen to see the crash."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Stop the service first
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    # Run the app with offscreen display and faulthandler enabled
    test_script = r'''import os, sys, faulthandler
os.chdir(os.path.expanduser("~/anilag"))
sys.path.insert(0, ".")
faulthandler.enable()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
print("Starting app with faulthandler...")
from dotenv import load_dotenv
load_dotenv("config.env")
print("Config loaded, calling main...")
import main as app_main
print("main module loaded, calling main()...")
try:
    app_main.main()
except SystemExit:
    pass
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Exception: {e}")
'''

    sftp = client.open_sftp()
    with sftp.file("/tmp/pi_app_test.py", "w") as f:
        f.write(test_script)
    sftp.close()

    stdin, stdout, stderr = client.exec_command(
        "cd ~/anilag && source venv/bin/activate && timeout 30 python /tmp/pi_app_test.py 2>&1",
        timeout=45
    )
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    print("STDOUT:", out)
    if err:
        print("STDERR:", err)

    client.close()

if __name__ == "__main__":
    main()
