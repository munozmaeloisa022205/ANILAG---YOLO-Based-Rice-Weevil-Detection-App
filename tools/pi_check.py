#!/usr/bin/env python3
"""SSH into the Pi and inspect the deployed detection model state."""
import paramiko
import sys

HOST = "anilag.local"
USER = "user"
PASSWORD = "user"

def run(client, cmd, label=None):
    if label:
        print(f"\n=== {label} ===")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
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
    try:
        print(f"Connecting to {USER}@{HOST}...")
        client.connect(HOST, username=USER, password=PASSWORD, timeout=15)
        print("Connected.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    # Service status
    run(client, "systemctl is-active anilag.service", "Service status")
    run(client, "systemctl is-enabled anilag.service", "Auto-start")

    # Config: detection-related settings
    run(client, "grep -E 'CONFIDENCE_THRESHOLD|IOU_THRESHOLD|YOLO_TILES|YOLO_TILE_MERGE|YOLO_CLAHE|USE_NCNN|YOLO_MAX_DET|YOLO_MIN_BOX|YOLO_MAX_BOX|YOLO_MAX_ASPECT|DETECTION_INTERVAL|YOLO_IMGSZ|MODEL_PATH|STORE_PREPROCESSED' ~/anilag/config.env", "Detection config")

    # Check if the _merge_boxes fix is deployed (the bug was self._merge_iou)
    run(client, "grep -n 'iou > iou_threshold\\|self._merge_iou\\|self._merge_containment' ~/anilag/src/detection/yolov11_detector.py", "Merge box fix check (should show iou_threshold, NOT self._merge)")

    # Check if diagnostic logging is deployed
    run(client, "grep -n 'DETECT\\|INFER\\|_detect_calls' ~/anilag/src/detection/yolov11_detector.py | head -20", "Diagnostic logging check")

    # Check the deployed detect() method
    run(client, "grep -n 'def detect\\|def _infer\\|def _merge_boxes\\|def _preprocess\\|def preprocess_frame' ~/anilag/src/detection/yolov11_detector.py", "Key methods")

    # Check model files
    run(client, "ls -la ~/anilag/models/ 2>/dev/null; ls -la ~/anilag/models/sitophilus_oryzae_v2-3_best_ncnn_model/ 2>/dev/null | head -10", "Model files")

    # Recent service logs (last 50 lines, looking for detection output)
    run(client, "sudo journalctl -u anilag.service -n 80 --no-pager 2>/dev/null || journalctl -u anilag.service -n 80 --no-pager 2>/dev/null", "Recent service logs")

    # Check if the app is running and what Python/version
    run(client, "python3 --version; pip3 list 2>/dev/null | grep -iE 'ultralytics|opencv|ncnn|torch'", "Python environment")

    # Check the NCNN metadata
    run(client, "cat ~/anilag/models/sitophilus_oryzae_v2-3_best_ncnn_model/metadata.yaml 2>/dev/null | head -20", "NCNN metadata")

    client.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
