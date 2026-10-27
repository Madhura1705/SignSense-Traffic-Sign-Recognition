#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "========================================================"
echo "  SignSense: Traffic Sign Recognition Web Server"
echo "========================================================"

if [ -f ".venv/bin/activate" ]; then
    echo "Activating Python virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating Python virtual environment (venv)..."
    source venv/bin/activate
fi

echo "Starting SignSense server on http://127.0.0.1:7860 ..."
python server.py "$@"
