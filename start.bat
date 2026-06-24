@echo off
REM Startup script for Anilag on Windows (for testing)

echo Starting Anilag - Rice Weevil Detection System...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if needed
pip install -r requirements.txt

REM Create necessary directories
if not exist "models" mkdir models
if not exist "logs" mkdir logs

REM Run the application
python main.py

pause
