#!/bin/bash
# Setup auto-start for Anilag on Raspberry Pi 5 boot.
#
# Usage:
#   cd ~/anilag
#   chmod +x deploy/setup_autostart.sh
#   ./deploy/setup_autostart.sh
#
# This script:
#   1. Installs a systemd service that launches Anilag on boot.
#   2. Enables the service so it starts automatically every time the Pi boots.
#   3. Optionally starts it right now (without rebooting).
#
# The service waits for the X display to be ready before launching the GUI,
# so it works even when the Pi boots to desktop with auto-login.

set -e

SERVICE_NAME="anilag.service"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_SRC="${APP_DIR}/deploy/${SERVICE_NAME}"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}"

echo "=== Anilag Auto-Start Setup ==="
echo "App directory: ${APP_DIR}"
echo ""

# --- Sanity checks ---

if [ "$(id -un)" != "user" ] && [ "$(id -un)" != "pi" ]; then
    echo "NOTE: This script is designed for the 'user' account on the Pi."
    echo "Current user is '$(id -un)'. The service file will need to be"
    echo "edited if this is not the intended user."
    echo ""
fi

if [ ! -f "${APP_DIR}/main.py" ]; then
    echo "ERROR: main.py not found in ${APP_DIR}"
    echo "Make sure you are running this from the anilag project directory."
    exit 1
fi

if [ ! -f "${SERVICE_SRC}" ]; then
    echo "ERROR: Service template not found at ${SERVICE_SRC}"
    exit 1
fi

# --- Install xdotool (used to hide the taskbar before launch) ---

if ! command -v xdotool >/dev/null 2>&1; then
    echo "Installing xdotool (used to hide the taskbar for full-screen mode)..."
    sudo apt install -y xdotool || echo "WARNING: xdotool install failed; taskbar hiding may not work."
fi

# --- Install the service ---

echo "Installing systemd service..."
sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"

# Replace the placeholder app directory and user with actual values.
# This lets the same service file work whether the app is at ~/anilag or
# somewhere else.
sudo sed -i "s|/home/user/anilag|${APP_DIR}|g" "${SERVICE_DST}"

CURRENT_USER="$(id -un)"
sudo sed -i "s/^User=user$/User=${CURRENT_USER}/" "${SERVICE_DST}"
sudo sed -i "s|/home/user/.Xauthority|/home/${CURRENT_USER}/.Xauthority|g" "${SERVICE_DST}"

# --- Reload systemd and enable ---

echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling ${SERVICE_NAME} to start on boot..."
sudo systemctl enable "${SERVICE_NAME}"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "The Anilag service will start automatically on every boot."
echo ""
echo "Commands:"
echo "  Start now:      sudo systemctl start ${SERVICE_NAME}"
echo "  Stop:           sudo systemctl stop ${SERVICE_NAME}"
echo "  Status:         sudo systemctl status ${SERVICE_NAME}"
echo "  View logs:      sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Disable autostart: sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "To start it right now without rebooting:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "Or reboot the Pi:"
echo "  sudo reboot"
