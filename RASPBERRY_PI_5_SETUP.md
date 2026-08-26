# Raspberry Pi 5 Setup Guide for Anilag

## Hardware Specifications
- **Device**: Raspberry Pi 5 (8GB RAM)
- **Storage**: KINGSTON A400 SSD 240GB (External)
- **Model**: YOLOv11n for Sitophilus Oryzae detection
- **Cameras**: Up to 2 USB webcams (dual camera support)

## SSD Setup Instructions

### 1. Mount the External SSD
```bash
# Identify the SSD device
lsblk

# Create mount point
sudo mkdir -p /mnt/ssd

# Mount the SSD (replace sda1 with your device)
sudo mount /dev/sda1 /mnt/ssd

# Make mount permanent (add to /etc/fstab)
echo "/dev/sda1 /mnt/ssd ext4 defaults,noatime 0 0" | sudo tee -a /etc/fstab

# Create Anilag directories
sudo mkdir -p /mnt/ssd/anilag/scans
sudo mkdir -p /mnt/ssd/anilag/database
sudo mkdir -p /mnt/ssd/anilag/logs
sudo chown -R pi:pi /mnt/ssd/anilag
```

### 2. Configure Anilag for SSD Storage
Copy `config.env.example` to `config.env` and update:
```bash
cp config.env.example config.env
```

Edit `config.env` with the SSD paths:
```
SCAN_STORAGE_PATH=/mnt/ssd/anilag/scans
DB_STORAGE_PATH=/mnt/ssd/anilag/database
LOG_FILE=/mnt/ssd/anilag/logs/detection_log.csv
```

## Performance Optimizations

### 1. Enable USB 3.0 Performance
The Raspberry Pi 5 supports USB 3.0 for maximum SSD performance:
- Connect SSD to blue USB 3.0 ports (not USB 2.0 ports)
- Ensure proper power supply (5V 5A recommended)
- For dual cameras, connect cameras to separate USB 3.0 ports for optimal bandwidth

### 2. Dual Camera Setup
For dual USB camera configuration:
```bash
# Identify connected cameras
ls /dev/video*

# Test camera connections
v4l2-ctl --list-devices

# Update config.env for dual camera mode
DUAL_CAMERA_MODE=true
LEFT_CAMERA_ID=0
RIGHT_CAMERA_ID=1
CAMERA_AUTO_DETECT=true
CAMERA_FOURCC=MJPG
```

**Important — `/dev/video` numbering:** each UVC webcam registers *two* nodes on
Raspberry Pi OS: one that streams video and one that only carries UVC metadata.
Two cameras therefore appear as `/dev/video0` … `/dev/video3`, where only 0 and 2
are real capture devices. The obvious `LEFT_CAMERA_ID=0` / `RIGHT_CAMERA_ID=1`
pairing would point the right camera at camera 0's metadata node, which never
produces an image. With `CAMERA_AUTO_DETECT=true` the app probes every node at
startup and remaps the IDs onto the ones that actually deliver frames, logging:
```
Capture-capable video devices: [0, 2]
Remapping right camera 1 -> /dev/video2 (device 1 does not stream video)
```

**USB bandwidth:** both cameras share one USB controller. Two uncompressed YUYV
streams at 640x480x30 exceed the isochronous bandwidth it will allocate, and the
second camera fails with `VIDIOC_STREAMON: No space left on device`. The default
`CAMERA_FOURCC=MJPG` makes the cameras compress on-board so both streams fit.

### 3. CPU and Memory Configuration
These are the values actually shipped in `config.env`:
- **Image size**: 640px (`YOLO_IMGSZ`) — matches what the weights were trained at
- **Device**: CPU inference (the Pi 5 has no CUDA GPU)
- **Confidence threshold**: 0.7 (raised above the 0.5 training default to suppress false positives)
- **IOU threshold**: 0.7 (from AI-MODEL training)
- **Max detections**: 300 (`YOLO_MAX_DET`, the training/val default)
- **Detection interval**: 200ms (`DETECTION_INTERVAL_MS`) so a cycle cannot monopolise the CPU
- **Inference threads**: 3 of the Pi 5's 4 cores (`THREAD_COUNT` blank = auto)

RAM is not the constraint on an 8GB Pi 5 — YOLOv11n needs well under 1GB. The
limit is the four CPU cores, which is what `THREAD_COUNT` and
`DETECTION_INTERVAL_MS` exist to manage.

If the frame rate is too low, reduce `YOLO_IMGSZ` (e.g. to 320) and re-export the
NCNN model at the same size — the app warns and falls back to PyTorch if the two
disagree. Expect lower recall on small weevils below 640px.

### 4. Model Loading Optimization
The YOLOv11n model is optimized for Raspberry Pi 5:
- Model size: ~5.2MB (sitophilus_oryzae_v2-3_best.pt)
- Fast loading from SSD
- Optimized for dual camera simultaneous processing

### 5. NCNN Model Export (Recommended for Pi 5)
NCNN is a neural network inference framework optimized for ARM CPUs. It provides
~4x faster inference compared to PyTorch on Raspberry Pi 5.

The application automatically detects and uses the NCNN model if available.
To export the model to NCNN format:

```bash
# On your development machine (not the Pi), run:
python -c "from ultralytics import YOLO; YOLO('models/sitophilus_oryzae_v2-3_best.pt').export(format='ncnn', imgsz=640)"

# This creates: models/sitophilus_oryzae_v2-3_best_ncnn_model/
# Copy this directory to the Pi alongside the .pt model file
```

**The export size must equal `YOLO_IMGSZ`.** The input resolution is baked into an
NCNN export, so the app compares the two and silently falls back to the much
slower PyTorch model when they differ (it prints a warning naming the correct
re-export command). With `YOLO_IMGSZ=640` in `config.env`, export at 640.

The NCNN model directory contains:
- `model.ncnn.bin` — model weights (binary)
- `model.ncnn.param` — model structure (text)
- `metadata.yaml` — class names and metadata

**Performance comparison (estimated on Pi 5, at 320px):**
- PyTorch: ~90ms/inference (~11 FPS)
- NCNN: ~22ms/inference (~45 FPS)

The shipped 640px setting processes four times as many pixels, so expect
proportionally longer inference times than the figures above. The live figure is
shown in the UI, and `DETECTION_INTERVAL_MS` caps the cycle rate regardless.

If NCNN fails to load, the app automatically falls back to PyTorch.

## Installation Steps

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install PyQt5 from apt (required)
PyQt5 publishes no `aarch64` wheel on PyPI, so `pip install PyQt5` on the Pi tries
to compile Qt from source and fails. Install the system package instead:
```bash
sudo apt install -y python3-pyqt5
```
`requirements.txt` skips PyQt5 automatically on `aarch64`.

### 3. Install Python Dependencies
The virtual environment must be created with `--system-site-packages` so it can
see the apt-installed PyQt5:
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```
On `aarch64` this installs `opencv-python-headless` rather than `opencv-python`.
The normal wheel bundles its own Qt libraries, and loading those next to the apt
PyQt5 puts two different Qt5 builds in one process, which aborts the app with
"Cannot mix incompatible Qt library". The app never calls `cv2.imshow`, so the
headless build loses nothing.

Or just run `./start.sh`, which performs all of the above and checks the
`video` group membership needed to open the cameras.

### 4. Install Hardware-Specific Dependencies
```bash
# LED control. On the Pi 5 the WS2813 strip is driven over hardware SPI via
# spidev - rpi_ws281x cannot work here (see the LED section below), and pip
# installs spidev automatically from requirements.txt.

# For 1-Wire temperature sensor support
sudo apt install -y python3-w1thermsensor
```

### 5. Enable Required Interfaces
```bash
sudo raspi-config
# Enable: I2C, SPI, Serial, 1-Wire

# Camera access requires membership of the 'video' group
sudo usermod -aG video $USER   # log out and back in afterwards
```

## Running Anilag

### Start the Application
```bash
python main.py
```

### Auto-Start on Boot (Raspberry Pi 5)

The application can launch automatically when the Pi boots so the operator does
not need to log in or type any commands. Two methods are provided — use **either**
the systemd service (recommended, runs as a managed service) **or** the desktop
autostart entry (runs inside the LXDE desktop session).

#### Method 1: systemd service (recommended)

```bash
# 1. Copy the service file into systemd
sudo cp scripts/anilag.service /etc/systemd/system/

# 2. Make the launch script executable
chmod +x scripts/anilag-autostart.sh

# 3. If Anilag is NOT installed at /home/pi/anilag, edit both files and
#    change every /home/pi/anilag path to your actual install directory:
#    - scripts/anilag-autostart.sh  (PROJECT_DIR and VENV_PYTHON)
#    - /etc/systemd/system/anilag.service (WorkingDirectory, ExecStart, Environment)

# 4. Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable anilag.service

# 5. Reboot to test
sudo reboot
```

After reboot the loading screen should appear automatically within a few seconds.
Check the service status and logs with:
```bash
sudo systemctl status anilag.service
sudo journalctl -u anilag.service -f
```

To disable auto-start:
```bash
sudo systemctl disable anilag.service
```

#### Method 2: desktop autostart entry

```bash
# 1. Make the launch script executable
chmod +x scripts/anilag-autostart.sh

# 2. Copy the .desktop file into the LXDE autostart directory
mkdir -p ~/.config/autostart
cp scripts/anilag.desktop ~/.config/autostart/

# 3. If Anilag is NOT installed at /home/pi/anilag, edit the .desktop file and
#    change the Exec= and Icon= paths to your actual install directory.

# 4. Reboot to test
sudo reboot
```

To disable, simply remove the file:
```bash
rm ~/.config/autostart/anilag.desktop
```

### Monitor Performance
```bash
# Check CPU usage
htop

# Check SSD performance
iostat -x 1

# Check memory usage
free -h
```

## Troubleshooting

### SSD Not Mounting
```bash
# Check disk status
sudo fdisk -l

# Format if needed (WARNING: destroys data)
sudo mkfs.ext4 /dev/sda1
```

### Model Loading Issues
- Ensure model file exists: `models/sitophilus_oryzae_v2-3_best.pt`
- Check file permissions: `ls -la models/`
- Verify ultralytics version: `pip show ultralytics`

### Performance Issues
- Ensure SSD is connected to USB 3.0 port
- Check thermal throttling: `vcgencmd measure_temp`
- Reduce camera resolution if needed in config.env

## Expected Performance
With Raspberry Pi 5 (8GB RAM) and KINGSTON A400 SSD:
- **Model loading**: < 2 seconds (NCNN), < 3 seconds (PyTorch fallback)
- **Single camera (NCNN)**: 30-45 FPS
- **Single camera (PyTorch)**: 10-15 FPS
- **Dual camera (NCNN)**: 15-22 FPS per camera (simultaneous)
- **Dual camera (PyTorch)**: 5-8 FPS per camera (simultaneous)
- **Detection accuracy**: High (trained model)
- **Storage speed**: Fast SSD I/O for logging and image capture
- **USB bandwidth**: Optimized for dual camera streaming via USB 3.0

## Dual Camera Mode Configuration
To enable dual camera support:
1. Connect two USB webcams to separate USB 3.0 ports
2. Set `DUAL_CAMERA_MODE=true` in config.env
3. Configure camera IDs (typically 0 and 1)
4. Restart the application

The system will display both camera feeds side-by-side with real-time detection on both streams.

## Maintenance
- Regular SSD health checks: `sudo smartctl -a /dev/sda1`
- Monitor disk space: `df -h /mnt/ssd`
- Backup detection logs periodically
- Update software regularly
