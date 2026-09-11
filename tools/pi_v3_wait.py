#!/usr/bin/env python3
"""Wait for v3 model to fully load and verify."""
import paramiko
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected. Waiting 60s for model to fully load...")
    time.sleep(60)

    stdin, stdout, stderr = client.exec_command("sudo systemctl is-active anilag.service 2>&1")
    print("Status:", stdout.read().decode().strip())

    stdin, stdout, stderr = client.exec_command(
        "sudo journalctl -u anilag.service -n 20 --no-pager 2>&1",
        timeout=30
    )
    print("\nLogs:")
    print(stdout.read().decode(errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
