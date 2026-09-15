#!/usr/bin/env python3
"""
FastAPI Backend Server for VisionFPV Web Benchmark Studio
Handles synthetic dataset generation, custom video upload, canvas BBox ROI selection,
and benchmark execution with Raspberry Pi 4 CPU throttling simulation.
"""

import os
import sys
import json
import subprocess
import shutil
import base64
import time
import cv2
import numpy as np
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

WEB_DIR = os.path.join(BASE_DIR, "web")
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")

from benchmark.synthetic_generator import generate_custom_synthetic_dataset, OUTPUT_VIDEO, OUTPUT_GT
from benchmark.benchmark_py_vs_cpp import run_benchmark, SUMMARY_PATH, CSV_PATH, PLOT_PATH, VIS_PATH

app = FastAPI(title="VisionFPV Web Benchmark Studio", version="1.0.0")

# Mount Static Data Files & Assets
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


class SyntheticGenRequest(BaseModel):
    num_frames: int = 300
    motion_pattern: str = "zigzag"
    occlusion_pct: float = 100.0
    shake_level: str = "medium"
    blur_level: str = "light"
    start_x: int = 80
    start_y: int = 150


class BenchmarkRunRequest(BaseModel):
    video_source: str = "synthetic"  # 'synthetic' or 'uploaded'
    custom_roi: Optional[List[int]] = None  # [x, y, w, h]
    raspi_simulation: bool = False  # CPUQuota=150%


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/frame0")
async def get_frame0(source: str = "synthetic"):
    """Get base64 image and dimensions of Frame 0 for interactive canvas drag ROI selection."""
    video_file = OUTPUT_VIDEO if source == "synthetic" else os.path.join(DATA_DIR, "uploaded_video.mp4")
    if not os.path.exists(video_file):
        video_file = OUTPUT_VIDEO
        if not os.path.exists(video_file):
            generate_custom_synthetic_dataset()

    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(status_code=400, detail="Could not read frame 0 from video.")

    h, w, c = frame.shape
    _, buffer = cv2.imencode(".jpg", frame)
    b64_str = base64.b64encode(buffer).decode("utf-8")

    # Read GT ROI if synthetic
    gt_roi = [50, 120, 60, 60]
    if os.path.exists(OUTPUT_GT):
        try:
            with open(OUTPUT_GT, "r") as f:
                gt_data = json.load(f)
                gt_roi = gt_data[0]["bbox"]
        except Exception:
            pass

    return {
        "width": w,
        "height": h,
        "image_b64": f"data:image/jpeg;base64,{b64_str}",
        "default_roi": gt_roi
    }


@app.post("/api/generate-synthetic")
async def api_generate_synthetic(req: SyntheticGenRequest):
    """Generate custom synthetic FPV dataset based on user parameters."""
    try:
        generate_custom_synthetic_dataset(
            num_frames=req.num_frames,
            motion_pattern=req.motion_pattern,
            occlusion_pct=req.occlusion_pct,
            shake_level=req.shake_level,
            blur_level=req.blur_level,
            start_x=req.start_x,
            start_y=req.start_y,
            output_video=OUTPUT_VIDEO,
            output_gt=OUTPUT_GT
        )
        return await get_frame0(source="synthetic")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-video")
async def api_upload_video(file: UploadFile = File(...)):
    """Upload custom video for tracking benchmark."""
    dest_path = os.path.join(DATA_DIR, "uploaded_video.mp4")
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return await get_frame0(source="uploaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")


@app.post("/api/run-benchmark")
async def api_run_benchmark(req: BenchmarkRunRequest):
    """Run benchmark simulation and return summary metrics, CSV data, and video URL."""
    try:
        cmd = ["python3", os.path.join(BASE_DIR, "benchmark", "benchmark_py_vs_cpp.py"), "--headless", "--save-video"]
        
        if req.raspi_simulation:
            cmd = ["systemd-run", "--user", "--scope", "-p", "CPUQuota=150%", "taskset", "-c", "0-3"] + cmd

        print(f"[API BENCHMARK] Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        with open(SUMMARY_PATH, "r") as f:
            summary = json.load(f)

        # Read CSV data for Chart.js
        csv_rows = []
        if os.path.exists(CSV_PATH):
            import pandas as pd
            df = pd.read_csv(CSV_PATH)
            csv_rows = df.to_dict(orient="records")

        return {
            "status": "success",
            "summary": summary,
            "chart_data": csv_rows,
            "plot_url": f"/data/py_vs_cpp_summary.png?t={int(time.time())}",
            "video_url": f"/data/py_vs_cpp_visualization.mp4?t={int(time.time())}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark run error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
