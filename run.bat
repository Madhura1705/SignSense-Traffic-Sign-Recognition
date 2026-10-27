@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   SignSense: Traffic Sign Recognition Web Server
echo ========================================================

if exist ".venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating Python virtual environment (venv)...
    call venv\Scripts\activate.bat
)

echo Starting SignSense server on http://127.0.0.1:7860 ...
python server.py %*
