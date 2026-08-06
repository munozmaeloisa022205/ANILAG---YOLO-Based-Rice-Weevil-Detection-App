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
```

### 3. Memory Optimization for 8GB RAM
The current configuration is optimized for 8GB RAM:
- **Image size**: 320px (balanced speed/accuracy)
- **Device**: CPU inference (appropriate for Pi 5)
- **Confidence threshold**: 0.5
- **IOU threshold**: 0.7 (from AI-MODEL training)
- **Max detections**: 100 (optimized for dual camera performance)
- **Augmentation disabled**: Faster inference for real-time video

### 4. Model Loading Optimization
The YOLOv11n model is optimized for Raspberry Pi 5:
- Model size: ~5.2MB (sitophilus_oryzae_v2-3_best.pt)
- Fast loading from SSD
- Efficient inference with 320px input size
- Optimized for dual camera simultaneous processing

### 5. NCNN Model Export (Recommended for Pi 5)
NCNN is a neural network inference framework optimized for ARM CPUs. It provides
~4x faster inference compared to PyTorch on Raspberry Pi 5.

The application automatically detects and uses the NCNN model if available.
To export the model to NCNN format:

```bash
# On your development machine (not the Pi), run:
python -c "from ultralytics import YOLO; YOLO('models/sitophilus_oryzae_v2-3_best.pt').export(format='ncnn', imgsz=320)"

# This creates: models/sitophilus_oryzae_v2-3_best_ncnn_model/
# Copy this directory to the Pi alongside the .pt model file
```

The NCNN model directory contains:
- `model.ncnn.bin` — model weights (binary)
- `model.ncnn.param` — model structure (text)
- `metadata.yaml` — class names and metadata

**Performance comparison (estimated on Pi 5):**
- PyTorch: ~90ms/inference (~11 FPS)
- NCNN: ~22ms/inference (~45 FPS)

If NCNN fails to load, the app automatically falls back to PyTorch.

## Installation Steps

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Hardware-Specific Dependencies
```bash
# For LED control (WS2812B)
pip install rpi_ws281x

# For 1-Wire temperature sensor support
sudo apt install -y python3-w1thermsensor
```

### 4. Enable Required Interfaces
```bash
sudo raspi-config
# Enable: I2C, SPI, Serial, 1-Wire
```

## Running Anilag

### Start the Application
```bash
python anilag.py
```

Or use the provided startup script:
```bash
./start.sh
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
