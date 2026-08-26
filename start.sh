#!/bin/bash
# Startup script for Anilag on Raspberry Pi

echo "Starting Anilag - Rice Weevil Detection System..."

# PyQt5 has no aarch64 wheel on PyPI, so on the Raspberry Pi it has to come from
# apt. Check for it before doing anything else - a missing PyQt5 otherwise only
# shows up as an ImportError after the long dependency install.
ARCH="$(uname -m)"
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "armv7l" ]; then
    if ! python3 -c "import PyQt5" 2>/dev/null; then
        echo "ERROR: PyQt5 is not installed system-wide (required on ${ARCH})."
        echo "Install it with: sudo apt install -y python3-pyqt5"
        exit 1
    fi
fi

# Check if virtual environment exists.
# --system-site-packages is required so the venv can see the apt-installed
# python3-pyqt5; an isolated venv on the Pi would have no usable Qt bindings.
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --system-site-packages venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -r requirements.txt

# Create necessary directories
mkdir -p models logs

# Reading /dev/video* requires membership of the 'video' group.
if [ -e /dev/video0 ] && ! id -nG | grep -qw video; then
    echo "WARNING: $(id -un) is not in the 'video' group; the cameras may not open."
    echo "Fix with: sudo usermod -aG video $(id -un)   (then log out and back in)"
fi

# Run the application
python main.py
