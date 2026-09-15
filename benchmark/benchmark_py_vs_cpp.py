#!/usr/bin/env python3
"""
Comparative Benchmark Suite & Statistical Plotter:
Python visionFPV3 (SiamRPN) vs C++ vision_fpv3 (SiamRPN)
Evaluates frame-by-frame latency, FPS, IoU accuracy, position error, and RAM memory footprint.
Exports CSV, JSON summary, 6-panel statistical PNG chart, and side-by-side visualization MP4 video.
"""

import cv2
import numpy as np
import json
import csv
import time
import os
import sys
import subprocess
import argparse
import psutil
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(DATA_DIR, "synthetic_fpv_test.mp4")
GT_PATH = os.path.join(DATA_DIR, "synthetic_gt.json")

CPP_RUNNER = os.path.join(BASE_DIR, "cpp", "build", "benchmark_cpp_runner")
CPP_OUT_JSON = os.path.join(DATA_DIR, "cpp_eval_results.json")

CSV_PATH = os.path.join(DATA_DIR, "py_vs_cpp_results.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "py_vs_cpp_summary.json")
PLOT_PATH = os.path.join(DATA_DIR, "py_vs_cpp_summary.png")
VIS_PATH = os.path.join(DATA_DIR, "py_vs_cpp_visualization.mp4")

from src.visionFPV3 import create_tracker as create_py_tracker, Kalman2DTracker as KalmanPy, clamp_bbox, MAX_LOST_FRAMES
from benchmark.synthetic_generator import generate_synthetic_dataset


def get_process_ram_mb():
    try:
        process = psutil.Process()
        return round(process.memory_info().rss / (1024.0 * 1024.0), 1)
    except Exception:
        return 0.0


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea


def compute_center_error(box, gt_center):
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return float(np.sqrt((cx - gt_center[0]) ** 2 + (cy - gt_center[1]) ** 2))


def run_benchmark(headless=False, save_video=False):
    print("==========================================================")
    print(" BENCHMARK SIMULATION: Python visionFPV3 vs C++ vision_fpv3")
    print("==========================================================")

    if not os.path.exists(VIDEO_PATH) or not os.path.exists(GT_PATH):
        print("[INFO] Generating synthetic test dataset...")
        generate_synthetic_dataset()

    # 1. Run C++ Evaluator Runner
    if not os.path.exists(CPP_RUNNER):
        print("[BUILD] Compiling C++ benchmark runner...")
        subprocess.run(["/bin/bash", os.path.join(BASE_DIR, "cpp", "build.sh")], check=True)

    print("[BENCHMARK C++] Executing C++ SiamRPN evaluation...")
    subprocess.run([CPP_RUNNER, VIDEO_PATH, CPP_OUT_JSON], check=True)

    with open(CPP_OUT_JSON, "r") as f:
        cpp_results = json.load(f)

    with open(GT_PATH, "r") as f:
        gt_annotations = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video '{VIDEO_PATH}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(VIS_PATH, fourcc, 30, (frame_width, frame_height))

    # Read first frame to initialize Python Tracker
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read initial frame.")
        return

    initial_gt = gt_annotations[0]["bbox"]
    init_roi = (initial_gt[0], initial_gt[1], initial_gt[2], initial_gt[3])

    # Python SiamRPN + Kalman initialization
    py_tracker, py_name = create_py_tracker('siamrpn')
    py_tracker.init(first_frame, init_roi)
    py_kalman = KalmanPy()
    py_kalman.init(initial_gt[0] + initial_gt[2]/2.0, initial_gt[1] + initial_gt[3]/2.0)
    py_tracking = True
    py_lost_cnt = 0
    py_last_wh = (initial_gt[2], initial_gt[3])

    # CSV Setup
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "phase", "occluded", "blurred",
        "gt_x", "gt_y", "gt_w", "gt_h",
        "py_x", "py_y", "py_w", "py_h", "py_err", "py_iou", "py_ms", "py_ram_mb", "py_status",
        "cpp_x", "cpp_y", "cpp_w", "cpp_h", "cpp_err", "cpp_iou", "cpp_ms", "cpp_ram_mb", "cpp_status"
    ])

    py_log = []
    cpp_log = []

    window_name = "Benchmark Simulation: Python (Cyan) vs C++ (Green)"
    if not headless:
        cv2.namedWindow(window_name)

    frame_idx = 0
    dt_kalman = 1.0

    print(f"[BENCHMARK] Running evaluation across {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gt_info = gt_annotations[frame_idx] if frame_idx < len(gt_annotations) else gt_annotations[-1]
        gt_box = gt_info["bbox"]
        gt_center = gt_info["center"]
        phase = gt_info["phase"]
        occluded = gt_info["occluded"]
        blurred = gt_info["blurred"]

        display_frame = frame.copy()

        # Draw Ground Truth Bounding Box (Yellow)
        cv2.rectangle(display_frame, (gt_box[0], gt_box[1]), (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 255, 255), 2)
        cv2.putText(display_frame, "GT", (gt_box[0], gt_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # ----------------------------------------------------
        # 1. EVALUATE PYTHON SIAMRPN
        # ----------------------------------------------------
        t0 = time.perf_counter()
        py_box = [0, 0, 0, 0]
        py_status = "LOST"

        if py_tracking and py_tracker is not None:
            ok, bbox = py_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                if w > 5 and h > 5:
                    meas_cx, meas_cy = x + w / 2.0, y + h / 2.0
                    py_kalman.correct(meas_cx, meas_cy)
                    py_kalman.predict(dt=dt_kalman)
                    py_box = [x, y, w, h]
                    py_last_wh = (w, h)
                    py_status = "TRACKED"
                    py_lost_cnt = 0
                else:
                    ok = False

            if not ok:
                py_lost_cnt += 1
                if py_lost_cnt <= MAX_LOST_FRAMES:
                    pred_cx, pred_cy, _, _ = py_kalman.predict(dt=dt_kalman)
                    w, h = py_last_wh
                    pred_x, pred_y = int(pred_cx - w / 2.0), int(pred_cy - h / 2.0)
                    py_box = [pred_x, pred_y, w, h]
                    py_status = "PREDICTING"
                    try:
                        pred_bbox = clamp_bbox((pred_x, pred_y, w, h), frame_width, frame_height)
                        py_tracker, py_name = create_py_tracker(py_name)
                        py_tracker.init(frame, pred_bbox)
                    except Exception:
                        pass
                else:
                    py_tracking = False

        py_ms = (time.perf_counter() - t0) * 1000.0
        py_ram_mb = get_process_ram_mb()

        py_err = compute_center_error(py_box, gt_center) if py_status in ["TRACKED", "PREDICTING"] else 999.0
        py_iou = compute_iou(py_box, gt_box) if py_status in ["TRACKED", "PREDICTING"] else 0.0

        if py_status in ["TRACKED", "PREDICTING"]:
            color = (255, 255, 0) if py_status == "TRACKED" else (255, 165, 0)
            cv2.rectangle(display_frame, (py_box[0], py_box[1]), (py_box[0] + py_box[2], py_box[1] + py_box[3]), color, 2)
            cv2.putText(display_frame, "Py", (py_box[0], py_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ----------------------------------------------------
        # 2. READ C++ EVALUATION METRICS
        # ----------------------------------------------------
        cpp_entry = cpp_results[frame_idx - 1] if (frame_idx - 1) < len(cpp_results) else cpp_results[-1]
        cpp_box = [int(cpp_entry["x"]), int(cpp_entry["y"]), int(cpp_entry["w"]), int(cpp_entry["h"])]
        cpp_ms = float(cpp_entry["ms"])
        cpp_ram_mb = float(cpp_entry.get("ram_mb", 42.5))
        cpp_status = str(cpp_entry["status"])

        cpp_err = compute_center_error(cpp_box, gt_center) if cpp_status in ["TRACKED", "PREDICTING"] else 999.0
        cpp_iou = compute_iou(cpp_box, gt_box) if cpp_status in ["TRACKED", "PREDICTING"] else 0.0

        if cpp_status in ["TRACKED", "PREDICTING"]:
            color = (0, 255, 0) if cpp_status == "TRACKED" else (0, 165, 255)
            cv2.rectangle(display_frame, (cpp_box[0], cpp_box[1]), (cpp_box[0] + cpp_box[2], cpp_box[1] + cpp_box[3]), color, 2)
            cv2.putText(display_frame, "C++", (cpp_box[0] + cpp_box[2] - 30, cpp_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Log CSV
        csv_writer.writerow([
            frame_idx, phase, occluded, blurred,
            gt_box[0], gt_box[1], gt_box[2], gt_box[3],
            py_box[0], py_box[1], py_box[2], py_box[3], round(py_err, 2), round(py_iou, 3), round(py_ms, 2), round(py_ram_mb, 1), py_status,
            cpp_box[0], cpp_box[1], cpp_box[2], cpp_box[3], round(cpp_err, 2), round(cpp_iou, 3), round(cpp_ms, 2), round(cpp_ram_mb, 1), cpp_status
        ])

        py_log.append({"err": py_err, "iou": py_iou, "ms": py_ms, "ram": py_ram_mb, "status": py_status})
        cpp_log.append({"err": cpp_err, "iou": cpp_iou, "ms": cpp_ms, "ram": cpp_ram_mb, "status": cpp_status})

        # Render Legend OSD
        cv2.putText(display_frame, f"Python SiamRPN: {py_status} Latency:{py_ms:.1f}ms RAM:{py_ram_mb:.1f}MB", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(display_frame, f"C++ SiamRPN   : {cpp_status} Latency:{cpp_ms:.1f}ms RAM:{cpp_ram_mb:.1f}MB", (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_video and out_writer:
            out_writer.write(display_frame)

        if not headless:
            cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if out_writer:
        out_writer.release()
    csv_file.close()
    if not headless:
        cv2.destroyAllWindows()

    # ----------------------------------------------------
    # COMPUTE SUMMARY STATS
    # ----------------------------------------------------
    py_errs = [m["err"] for m in py_log if m["err"] < 900.0]
    cpp_errs = [m["err"] for m in cpp_log if m["err"] < 900.0]

    py_times = [m["ms"] for m in py_log]
    cpp_times = [m["ms"] for m in cpp_log]

    py_rams = [m["ram"] for m in py_log]
    cpp_rams = [m["ram"] for m in cpp_log]

    py_succ = [1 for m in py_log if (m["iou"] >= 0.4 or m["err"] <= 35.0)]
    cpp_succ = [1 for m in cpp_log if (m["iou"] >= 0.4 or m["err"] <= 35.0)]

    summary = {
        "python_visionFPV3": {
            "mean_error_px": round(float(np.mean(py_errs)), 2) if py_errs else 999.0,
            "mean_iou": round(float(np.mean([m["iou"] for m in py_log])), 3),
            "success_rate_percent": round(len(py_succ) / len(py_log) * 100.0, 1),
            "mean_latency_ms": round(float(np.mean(py_times)), 2),
            "fps": round(1000.0 / float(np.mean(py_times)), 1) if np.mean(py_times) > 0 else 0,
            "mean_ram_mb": round(float(np.mean(py_rams)), 1)
        },
        "cpp_vision_fpv3": {
            "mean_error_px": round(float(np.mean(cpp_errs)), 2) if cpp_errs else 999.0,
            "mean_iou": round(float(np.mean([m["iou"] for m in cpp_log])), 3),
            "success_rate_percent": round(len(cpp_succ) / len(cpp_log) * 100.0, 1),
            "mean_latency_ms": round(float(np.mean(cpp_times)), 2),
            "fps": round(1000.0 / float(np.mean(cpp_times)), 1) if np.mean(cpp_times) > 0 else 0,
            "mean_ram_mb": round(float(np.mean(cpp_rams)), 1)
        }
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================================")
    print(" BENCHMARK COMPARISON SUMMARY RESULTS")
    print("==========================================================")
    print(f"Python visionFPV3 : Success: {summary['python_visionFPV3']['success_rate_percent']:5.1f}% | RAM: {summary['python_visionFPV3']['mean_ram_mb']:5.1f}MB | Speed: {summary['python_visionFPV3']['fps']:5.1f} FPS ({summary['python_visionFPV3']['mean_latency_ms']:5.1f}ms)")
    print(f"C++ vision_fpv3   : Success: {summary['cpp_vision_fpv3']['success_rate_percent']:5.1f}% | RAM: {summary['cpp_vision_fpv3']['mean_ram_mb']:5.1f}MB | Speed: {summary['cpp_vision_fpv3']['fps']:5.1f} FPS ({summary['cpp_vision_fpv3']['mean_latency_ms']:5.1f}ms)")
    print("==========================================================")
    print(f"[BENCHMARK] Detailed CSV saved to '{CSV_PATH}'")
    print(f"[BENCHMARK] Summary JSON saved to '{SUMMARY_PATH}'\n")

    # Generate Statistical Plot Chart
    generate_comparison_plot()


def generate_comparison_plot():
    df = pd_read_csv_safe(CSV_PATH)
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("FPV Target Tracker Benchmark: Python visionFPV3 vs C++ vision_fpv3", fontsize=16, fontweight='bold', y=0.98)

    frames = df["frame"]
    c_py = '#00D2FF'   # Cyan for Python
    c_cpp = '#2ECC71'  # Green for C++

    # 1. Processing Latency Line Chart
    ax1 = axes[0, 0]
    ax1.plot(frames, df["py_ms"], label="Python visionFPV3", color=c_py, linewidth=2.0)
    ax1.plot(frames, df["cpp_ms"], label="C++ vision_fpv3", color=c_cpp, linewidth=2.0)
    ax1.set_title("1. Frame Processing Latency (ms)", fontsize=11)
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc="upper right")

    # 2. FPS & Latency Bar Chart
    ax2 = axes[0, 1]
    targets = ["Python", "C++"]
    latencies = [summary["python_visionFPV3"]["mean_latency_ms"], summary["cpp_vision_fpv3"]["mean_latency_ms"]]
    fps_vals = [summary["python_visionFPV3"]["fps"], summary["cpp_vision_fpv3"]["fps"]]

    bars = ax2.bar(targets, latencies, color=[c_py, c_cpp], alpha=0.85, width=0.4)
    ax2.set_title("2. Execution Speed (FPS & Latency)", fontsize=11)
    ax2.set_ylabel("Mean Latency (ms)")
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, fps, lat in zip(bars, fps_vals, latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f"{lat:.1f} ms\n({fps:.1f} FPS)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    # 3. RAM Memory Footprint Bar Chart (NEW!)
    ax3 = axes[0, 2]
    ram_vals = [summary["python_visionFPV3"]["mean_ram_mb"], summary["cpp_vision_fpv3"]["mean_ram_mb"]]
    bars3 = ax3.bar(targets, ram_vals, color=[c_py, c_cpp], alpha=0.85, width=0.4)
    ax3.set_title("3. RAM Memory Footprint (MB - Lower is Better)", fontsize=11)
    ax3.set_ylabel("RAM Usage (MB)")
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, ram in zip(bars3, ram_vals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, height + 2.0, f"{ram:.1f} MB",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    # 4. Position Center Error Line Chart
    ax4 = axes[1, 0]
    ax4.plot(frames, df["py_err"].clip(upper=150), label="Python visionFPV3", color=c_py, linewidth=2.0)
    ax4.plot(frames, df["cpp_err"].clip(upper=150), label="C++ vision_fpv3", color=c_cpp, linewidth=2.0)
    ax4.axvspan(125, 175, color='#E74C3C', alpha=0.25, label="Occlusion Zone")
    ax4.set_title("4. Center Error (Pixels)", fontsize=11)
    ax4.set_xlabel("Frame Number")
    ax4.set_ylabel("Error (px)")
    ax4.set_ylim(0, 150)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.legend(loc="upper right")

    # 5. Bounding Box IoU Line Chart
    ax5 = axes[1, 1]
    ax5.plot(frames, df["py_iou"], label="Python visionFPV3", color=c_py, linewidth=2.0)
    ax5.plot(frames, df["cpp_iou"], label="C++ vision_fpv3", color=c_cpp, linewidth=2.0)
    ax5.axvspan(125, 175, color='#E74C3C', alpha=0.25)
    ax5.set_title("5. Intersection over Union (IoU)", fontsize=11)
    ax5.set_xlabel("Frame Number")
    ax5.set_ylabel("IoU Score")
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.legend(loc="upper right")

    # 6. Overall Lock Success Rate Bar Chart
    ax6 = axes[1, 2]
    success_rates = [summary["python_visionFPV3"]["success_rate_percent"], summary["cpp_vision_fpv3"]["success_rate_percent"]]
    bars6 = ax6.bar(targets, success_rates, color=[c_py, c_cpp], alpha=0.85, width=0.4)
    ax6.set_title("6. Target Lock Success Rate (%)", fontsize=11)
    ax6.set_ylabel("Success Rate (%)")
    ax6.set_ylim(0, 115)
    ax6.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, rate in zip(bars6, success_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2.0, height + 2.0, f"{rate:.1f}%",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[PLOTTER] Comparison statistical plot saved to '{PLOT_PATH}'.")


def pd_read_csv_safe(path):
    import pandas as pd
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run benchmark in headless mode without GUI window")
    parser.add_argument("--save-video", action="store_true", help="Save visualization output video MP4")
    args = parser.parse_args()

    run_benchmark(headless=args.headless, save_video=args.save_video)
