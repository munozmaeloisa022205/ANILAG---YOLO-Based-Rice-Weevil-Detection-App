#!/usr/bin/env python3
"""Check config.env on Pi."""
import paramiko
import sys

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    stdin, stdout, stderr = client.exec_command("grep MODEL_PATH ~/anilag/config.env 2>&1")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    stdin, stdout, stderr = client.exec_command("ls -la ~/anilag/models/sitophilus_oryzae_v3* 2>&1")
    print("\nV3 model files:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    stdin, stdout, stderr = client.exec_command("ls -la ~/anilag/models/sitophilus_oryzae_v3_best_ncnn_model/ 2>&1")
    print("\nV3 NCNN dir:")
    out = stdout.read().decode(errors="replace")
    sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))

    client.close()

if __name__ == "__main__":
    main()
