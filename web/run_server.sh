#!/usr/bin/env bash
# VisionFPV Benchmark Studio Launcher
# Starts FastAPI + Uvicorn Web Server on http://localhost:8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASE_DIR"

echo "=========================================================="
echo " 🎯 VisionFPV Web Benchmark Studio"
echo " Backend Server Starting on http://localhost:8000"
echo "=========================================================="

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"
python3 -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
