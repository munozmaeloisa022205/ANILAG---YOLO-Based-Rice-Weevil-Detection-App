#!/usr/bin/env python3
"""Reset service and restart cleanly."""
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

    # Stop the service, wait, then reset and start
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    print("Stop:", stdout.read().decode().strip())
    time.sleep(5)

    # Reset failure counter
    stdin, stdout, stderr = client.exec_command("sudo systemctl reset-failed anilag.service 2>&1")
    stdout.read()
    time.sleep(2)

    # Start the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl start anilag.service 2>&1")
    stdout.read()
    time.sleep(40)

    # Check status
    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    status = stdout.read().decode().strip()
    print("Status:", status)

    # Check recent logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '1 min ago' --no-pager 2>&1 | tail -20",
        timeout=30
    )
    print("\nLogs:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
