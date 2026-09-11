# Anilag - Rice Weevil Detection System

A PyQt5-based GUI application for detecting rice weevils using YOLOv11n on Raspberry Pi 5 with USB webcams.

## Features

- **Live Camera Feed**: Real-time detection from USB webcams
- **YOLOv11n Detection**: Optimized rice weevil detection model
- **Smart Logging**: Automatic logging with action recommendations
- **Email Notifications**: Email alerts on every detection activity
- **Recommendation System**:
  - "Heat Treatment Needed" - when rice weevils are detected after mixing/sifting
  - "Take Out Rice" - when no rice weevils detected after mixing/sifting

## Hardware Requirements

- Raspberry Pi 5
- USB webcams (left and right)

## Software Requirements

- Python 3.9+
- PyQt5
- ultralytics (YOLOv11n)
- OpenCV

## Installation

1. Clone the repository:
```bash
cd /path/to/anilag
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
Edit `config.env` with your email credentials and hardware configurations.

4. Enable camera:
```bash
sudo raspi-config
# Navigate to Interface Options -> Camera -> Enable
```

5. Download YOLOv11n model:
```bash
mkdir -p models
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt').save('models/yolov11n.pt')"
```

## Usage

Run the application:
```bash
python main.py
```

### GUI Controls

- **Start Scan**: Begin live detection with camera feed
- **Stop Scan**: Stop detection

### Detection Workflow

1. Click "Start Scan" to begin detection
2. The system will:
   - Display live camera feed with detection boxes
   - Log all detections with timestamps
   - Generate recommendations based on detection results
   - Send email notifications for each activity

## Training Custom Model (Optional)

To train a custom YOLOv11n model for rice weevil detection:

1. Prepare your dataset in YOLO format
2. Update dataset configuration
3. Run training:
```bash
python train_model.py --data your_dataset.yaml --epochs 100
```

## Troubleshooting

### Camera not detected
- Ensure camera is enabled in raspi-config
- Check camera cable connection
- Test with `libcamera-hello`

### Email not sending
- Verify SMTP settings
- Use App Password for Gmail (not regular password)
- Check network connection

## Project Structure

```
anilag/
├── main.py                 # Application entry point
├── config.env              # Configuration file
├── requirements.txt        # Python dependencies
├── models/                 # YOLO models
│   └── yolov11n.pt
├── logs/                   # Detection logs
│   └── detection_log.csv
├── src/
│   ├── __init__.py
│   ├── gui/               # GUI components
│   │   ├── __init__.py
│   │   └── main_window.py
│   ├── detection/         # Detection module
│   │   ├── __init__.py
│   │   └── yolov11_detector.py
│   ├── hardware/          # Hardware interfaces
│   │   ├── __init__.py
│   │   └── camera.py
│   ├── logging/           # Logging system
│   │   ├── __init__.py
│   │   └── logger.py
│   └── notification/      # Email notifications
│       ├── __init__.py
│       └── email_notifier.py
└── README.md
```

## License

MIT License
