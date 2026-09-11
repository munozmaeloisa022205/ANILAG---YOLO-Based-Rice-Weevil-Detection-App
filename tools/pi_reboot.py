#!/usr/bin/env python3
"""Clear model pycache and reboot Pi."""
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

    # Stop the service
    stdin, stdout, stderr = client.exec_command("sudo systemctl stop anilag.service 2>&1")
    stdout.read()
    time.sleep(3)

    # Clear model __pycache__
    stdin, stdout, stderr = client.exec_command(
        "rm -rf ~/anilag/models/sitophilus_oryzae_v2-3_best_ncnn_model/__pycache__ 2>&1; echo done"
    )
    print("Clear model pycache:", stdout.read().decode().strip())

    # Clear all pycache
    stdin, stdout, stderr = client.exec_command(
        "find ~/anilag -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; echo done"
    )
    print("Clear all pycache:", stdout.read().decode().strip())

    # Reboot the Pi
    print("Rebooting Pi...")
    stdin, stdout, stderr = client.exec_command("sudo reboot 2>&1")
    stdout.read()
    client.close()

    # Wait for Pi to come back
    print("Waiting 60s for reboot...")
    time.sleep(60)

    # Reconnect
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(5):
        try:
            client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
            print("Reconnected!")
            break
        except:
            print(f"Retry {attempt+1}...")
            time.sleep(10)

    # Check if service is running
    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    status = stdout.read().decode().strip()
    print("Service status:", status)

    # Check logs
    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service --since '2 min ago' --no-pager 2>&1 | tail -20",
        timeout=30
    )
    print("\nLogs:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
