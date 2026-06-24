#!/bin/bash
# Startup script for Anilag on Raspberry Pi

echo "Starting Anilag - Rice Weevil Detection System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Create necessary directories
mkdir -p models logs

# Run the application
python main.py
