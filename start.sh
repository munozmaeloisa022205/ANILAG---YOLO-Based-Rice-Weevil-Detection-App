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
        echo "Install it with: sudo apt install -y python3-pyqt5 python3-dev"
        echo "(python3-dev is also needed to build the spidev package)"
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
pip install -r requirements.txt || { echo "ERROR: dependency install failed."; exit 1; }

# ultralytics declares a hard dependency on the regular opencv-python wheel, and
# pip does not treat opencv-python-headless as satisfying it - so on the Pi it
# gets installed anyway. That wheel bundles its own Qt libraries, which clash
# with the apt PyQt5 ("Cannot mix incompatible Qt library" / xcb plugin abort).
# Remove it; opencv-python-headless from requirements.txt provides cv2.
if [ "${ARCH}" = "aarch64" ] || [ "${ARCH}" = "armv7l" ]; then
    if pip show opencv-python >/dev/null 2>&1; then
        echo "Removing non-headless opencv-python (conflicts with apt PyQt5)..."
        pip uninstall -y opencv-python
    fi
fi

# Create necessary directories
mkdir -p models logs

# The fine-tuned weevil model is not in the git repo and has to be copied over
# manually. Without it the app falls back to warning about a generic COCO model
# and suppresses all detections, which looks like "nothing is ever detected".
MODEL_FILE="$(grep -E '^MODEL_PATH=' config.env 2>/dev/null | cut -d= -f2)"
MODEL_FILE="${MODEL_FILE:-models/sitophilus_oryzae_v2-3_best.pt}"
if [ ! -e "${MODEL_FILE}" ]; then
    echo "WARNING: model file '${MODEL_FILE}' not found."
    echo "Copy the fine-tuned checkpoint onto this machine or set MODEL_PATH in config.env."
fi

# Reading /dev/video* requires membership of the 'video' group.
if [ -e /dev/video0 ] && ! id -nG | grep -qw video; then
    echo "WARNING: $(id -un) is not in the 'video' group; the cameras may not open."
    echo "Fix with: sudo usermod -aG video $(id -un)   (then log out and back in)"
fi

# Run the application
python main.py
