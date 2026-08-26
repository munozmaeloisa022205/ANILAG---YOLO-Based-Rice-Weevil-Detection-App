#!/bin/bash
# Anilag autostart launcher for Raspberry Pi 5.
#
# This script is used by both the systemd service and the LXDE autostart entry.
# It waits for the desktop environment to be ready, then launches the Anilag
# GUI application with the project's virtual environment.
#
# Install (see RASPBERRY_PI_5_SETUP.md -> "Auto-Start on Boot"):
#   sudo cp scripts/anilag.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable anilag.service

# --- Configuration -----------------------------------------------------------
# Edit these two paths to match your Pi installation.
PROJECT_DIR="/home/pi/anilag"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"

# --- Launch ------------------------------------------------------------------
cd "${PROJECT_DIR}" || exit 1

# Wait for the X server / desktop to be available. On a headless boot the
# framebuffer may not be ready for several seconds, and PyQt5 will crash
# if it tries to create a window before DISPLAY is set.
while [ -z "${DISPLAY}" ]; do
    sleep 1
done

# Small extra delay so the desktop has fully settled before we open a
# fullscreen window on top of it.
sleep 3

exec "${VENV_PYTHON}" "${PROJECT_DIR}/main.py"
