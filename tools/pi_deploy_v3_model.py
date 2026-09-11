#!/usr/bin/env python3
"""Deploy the new v3 model to the Pi and restart the service."""
import paramiko
import os
import time

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"
LOCAL_DIR = r"C:\Users\Eloisa\CascadeProjects\anilag"
REMOTE_DIR = "/home/user/anilag"

def run(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return out, err

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
    print("Connected.\n")

    # Stop the service first to free resources during upload
    print("Stopping service...")
    run(client, "sudo systemctl stop anilag.service 2>&1")
    time.sleep(3)

    sftp = client.open_sftp()

    # Create the remote NCNN model directory
    remote_ncnn_dir = f"{REMOTE_DIR}/models/sitophilus_oryzae_v3_best_ncnn_model"
    try:
        sftp.mkdir(remote_ncnn_dir)
    except IOError:
        pass  # already exists

    # Upload NCNN model files
    local_ncnn_dir = os.path.join(LOCAL_DIR, "models", "sitophilus_oryzae_v3_best_ncnn_model")
    ncnn_files = ["metadata.yaml", "model.ncnn.bin", "model.ncnn.param", "model_ncnn.py"]
    for fname in ncnn_files:
        local_path = os.path.join(local_ncnn_dir, fname)
        remote_path = f"{remote_ncnn_dir}/{fname}"
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {fname} ({size_mb:.1f} MB) -> {remote_path}")
        sftp.put(local_path, remote_path)

    # Upload the .pt model (for fallback/metadata)
    local_pt = os.path.join(LOCAL_DIR, "models", "sitophilus_oryzae_v3_best.pt")
    remote_pt = f"{REMOTE_DIR}/models/sitophilus_oryzae_v3_best.pt"
    size_mb = os.path.getsize(local_pt) / (1024 * 1024)
    print(f"Uploading sitophilus_oryzae_v3_best.pt ({size_mb:.1f} MB) -> {remote_pt}")
    sftp.put(local_pt, remote_pt)

    # Upload updated config.env
    local_config = os.path.join(LOCAL_DIR, "config.env")
    remote_config = f"{REMOTE_DIR}/config.env"
    print(f"Uploading config.env -> {remote_config}")
    sftp.put(local_config, remote_config)

    sftp.close()

    # Verify the model files are on the Pi
    out, err = run(client, f"ls -la {remote_ncnn_dir}/")
    print(f"\nRemote NCNN model files:\n{out}")

    # Verify config
    out, err = run(client, f"grep MODEL_PATH {REMOTE_DIR}/config.env")
    print(f"Config: {out}")

    # Start the service
    print("\nStarting service...")
    run(client, "sudo systemctl reset-failed anilag.service 2>&1")
    run(client, "sudo systemctl start anilag.service 2>&1")
    time.sleep(45)

    # Check status
    out, err = run(client, "sudo systemctl is-active anilag.service 2>&1")
    print(f"Status: {out}")

    # Check logs
    out, err = run(client, "sudo journalctl -u anilag.service --since '1 min ago' --no-pager 2>&1 | tail -20", timeout=30)
    print(f"\nLogs:\n{out}")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
