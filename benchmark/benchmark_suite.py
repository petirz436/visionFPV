#!/usr/bin/env python3
"""
FPV Tracker Benchmark Suite: V1 (CSRT) vs V2 (CSRT+Kalman) vs V3 (SiamRPN+Kalman)
Executes frame-by-frame evaluation against Ground Truth dataset and exports CSV metrics.
"""

import cv2
import numpy as np
import json
import csv
import time
import os
import argparse

import sys

# Set up project base path & data directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Import tracker implementations from src package
from src.visionFPV import create_csrt_tracker as create_v1_tracker
from src.visionFPV2 import create_csrt_tracker as create_v2_csrt, Kalman2DTracker as KalmanV2
from src.visionFPV3 import create_tracker as create_v3_tracker, Kalman2DTracker as KalmanV3, clamp_bbox, MAX_LOST_FRAMES

VIDEO_PATH = os.path.join(DATA_DIR, "synthetic_fpv_test.mp4")
GT_PATH = os.path.join(DATA_DIR, "synthetic_gt.json")
CSV_PATH = os.path.join(DATA_DIR, "benchmark_results.csv")
SUMMARY_PATH = os.path.join(DATA_DIR, "benchmark_summary.json")


def compute_iou(boxA, boxB):
    """Compute Intersection Over Union (IoU) of two bounding boxes (x, y, w, h)."""
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
    """Compute Euclidean distance (pixels) between box center and ground truth center."""
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return float(np.sqrt((cx - gt_center[0]) ** 2 + (cy - gt_center[1]) ** 2))


def run_benchmark(headless=False, save_video=False):
    print("==========================================================")
    print(" FPV TRACKER BENCHMARK SUITE (V1 vs V2 vs V3)")
    print("==========================================================")

    if not os.path.exists(VIDEO_PATH) or not os.path.exists(GT_PATH):
        print("[ERROR] Test dataset not found. Generating synthetic dataset first...")
        from benchmark.synthetic_generator import generate_synthetic_dataset
        generate_synthetic_dataset()

    with open(GT_PATH, "r") as f:
        gt_annotations = json.load(f)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video '{VIDEO_PATH}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Video Writer if requested
    out_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(os.path.join(DATA_DIR, "benchmark_visualization.mp4"), fourcc, 30, (frame_width, frame_height))

    # Read first frame to initialize all 3 trackers on Ground Truth ROI
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read initial frame.")
        return

    initial_gt = gt_annotations[0]["bbox"]
    init_roi = (initial_gt[0], initial_gt[1], initial_gt[2], initial_gt[3])

    # ----------------------------------------------------
    # TRACKER INSTANTIATION
    # ----------------------------------------------------
    # V1 (CSRT)
    t1_tracker = create_v1_tracker()
    t1_tracker.init(first_frame, init_roi)
    t1_tracking = True

    # V2 (CSRT + Kalman)
    t2_tracker = create_v2_csrt()
    t2_tracker.init(first_frame, init_roi)
    t2_kalman = KalmanV2()
    t2_kalman.init(initial_gt[0] + initial_gt[2]/2.0, initial_gt[1] + initial_gt[3]/2.0)
    t2_tracking = True
    t2_lost_cnt = 0
    t2_last_wh = (initial_gt[2], initial_gt[3])

    # V3 (SiamRPN + Kalman)
    t3_tracker, t3_name = create_v3_tracker('siamrpn')
    t3_tracker.init(first_frame, init_roi)
    t3_kalman = KalmanV3()
    t3_kalman.init(initial_gt[0] + initial_gt[2]/2.0, initial_gt[1] + initial_gt[3]/2.0)
    t3_tracking = True
    t3_lost_cnt = 0
    t3_last_wh = (initial_gt[2], initial_gt[3])

    # CSV Logging Setup
    csv_file = open(CSV_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "phase", "occluded", "blurred",
        "gt_x", "gt_y", "gt_w", "gt_h",
        "v1_x", "v1_y", "v1_w", "v1_h", "v1_err", "v1_iou", "v1_ms", "v1_status",
        "v2_x", "v2_y", "v2_w", "v2_h", "v2_err", "v2_iou", "v2_ms", "v2_status",
        "v3_x", "v3_y", "v3_w", "v3_h", "v3_err", "v3_iou", "v3_ms", "v3_status"
    ])

    metrics_log = {"v1": [], "v2": [], "v3": []}

    window_name = "FPV Tracker Benchmark Overlay"
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
        # 1. EVALUATE V1 (CSRT)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        v1_box = [0, 0, 0, 0]
        v1_status = "LOST"
        if t1_tracking and t1_tracker is not None:
            ok, bbox = t1_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                if w > 5 and h > 5:
                    v1_box = [x, y, w, h]
                    v1_status = "TRACKED"
                else:
                    t1_tracking = False
            else:
                t1_tracking = False

        v1_time_ms = (time.perf_counter() - t0) * 1000.0
        v1_err = compute_center_error(v1_box, gt_center) if v1_status == "TRACKED" else 999.0
        v1_iou = compute_iou(v1_box, gt_box) if v1_status == "TRACKED" else 0.0

        if v1_status == "TRACKED":
            cv2.rectangle(display_frame, (v1_box[0], v1_box[1]), (v1_box[0] + v1_box[2], v1_box[1] + v1_box[3]), (0, 0, 255), 2)

        # ----------------------------------------------------
        # 2. EVALUATE V2 (CSRT + KALMAN)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        v2_box = [0, 0, 0, 0]
        v2_status = "LOST"

        if t2_tracking and t2_tracker is not None:
            ok, bbox = t2_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                if w > 5 and h > 5:
                    meas_cx, meas_cy = x + w / 2.0, y + h / 2.0
                    t2_kalman.correct(meas_cx, meas_cy)
                    t2_kalman.predict(dt=dt_kalman)
                    v2_box = [x, y, w, h]
                    v2_last_wh = (w, h)
                    v2_status = "TRACKED"
                    t2_lost_cnt = 0
                else:
                    ok = False

            if not ok:
                t2_lost_cnt += 1
                if t2_lost_cnt <= MAX_LOST_FRAMES:
                    pred_cx, pred_cy, _, _ = t2_kalman.predict(dt=dt_kalman)
                    w, h = t2_last_wh
                    pred_x, pred_y = int(pred_cx - w / 2.0), int(pred_cy - h / 2.0)
                    v2_box = [pred_x, pred_y, w, h]
                    v2_status = "PREDICTING"
                    try:
                        pred_bbox = clamp_bbox((pred_x, pred_y, w, h), frame_width, frame_height)
                        t2_tracker = create_v2_csrt()
                        t2_tracker.init(frame, pred_bbox)
                    except Exception:
                        pass
                else:
                    t2_tracking = False

        v2_time_ms = (time.perf_counter() - t0) * 1000.0
        v2_err = compute_center_error(v2_box, gt_center) if v2_status in ["TRACKED", "PREDICTING"] else 999.0
        v2_iou = compute_iou(v2_box, gt_box) if v2_status in ["TRACKED", "PREDICTING"] else 0.0

        if v2_status in ["TRACKED", "PREDICTING"]:
            color = (0, 255, 0) if v2_status == "TRACKED" else (0, 165, 255)
            cv2.rectangle(display_frame, (v2_box[0], v2_box[1]), (v2_box[0] + v2_box[2], v2_box[1] + v2_box[3]), color, 2)

        # ----------------------------------------------------
        # 3. EVALUATE V3 (SiamRPN + KALMAN)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        v3_box = [0, 0, 0, 0]
        v3_status = "LOST"

        if t3_tracking and t3_tracker is not None:
            ok, bbox = t3_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                if w > 5 and h > 5:
                    meas_cx, meas_cy = x + w / 2.0, y + h / 2.0
                    t3_kalman.correct(meas_cx, meas_cy)
                    t3_kalman.predict(dt=dt_kalman)
                    v3_box = [x, y, w, h]
                    v3_last_wh = (w, h)
                    v3_status = "TRACKED"
                    t3_lost_cnt = 0
                else:
                    ok = False

            if not ok:
                t3_lost_cnt += 1
                if t3_lost_cnt <= MAX_LOST_FRAMES:
                    pred_cx, pred_cy, _, _ = t3_kalman.predict(dt=dt_kalman)
                    w, h = t3_last_wh
                    pred_x, pred_y = int(pred_cx - w / 2.0), int(pred_cy - h / 2.0)
                    v3_box = [pred_x, pred_y, w, h]
                    v3_status = "PREDICTING"
                    try:
                        pred_bbox = clamp_bbox((pred_x, pred_y, w, h), frame_width, frame_height)
                        t3_tracker, t3_name = create_v3_tracker(t3_name)
                        t3_tracker.init(frame, pred_bbox)
                    except Exception:
                        pass
                else:
                    t3_tracking = False

        v3_time_ms = (time.perf_counter() - t0) * 1000.0
        v3_err = compute_center_error(v3_box, gt_center) if v3_status in ["TRACKED", "PREDICTING"] else 999.0
        v3_iou = compute_iou(v3_box, gt_box) if v3_status in ["TRACKED", "PREDICTING"] else 0.0

        if v3_status in ["TRACKED", "PREDICTING"]:
            color = (255, 255, 0) if v3_status == "TRACKED" else (255, 165, 0)
            cv2.rectangle(display_frame, (v3_box[0], v3_box[1]), (v3_box[0] + v3_box[2], v3_box[1] + v3_box[3]), color, 2)

        # ----------------------------------------------------
        # LOG CSV & COLLECT METRICS
        # ----------------------------------------------------
        csv_writer.writerow([
            frame_idx, phase, occluded, blurred,
            gt_box[0], gt_box[1], gt_box[2], gt_box[3],
            v1_box[0], v1_box[1], v1_box[2], v1_box[3], round(v1_err, 2), round(v1_iou, 3), round(v1_time_ms, 2), v1_status,
            v2_box[0], v2_box[1], v2_box[2], v2_box[3], round(v2_err, 2), round(v2_iou, 3), round(v2_time_ms, 2), v2_status,
            v3_box[0], v3_box[1], v3_box[2], v3_box[3], round(v3_err, 2), round(v3_iou, 3), round(v3_time_ms, 2), v3_status,
        ])

        metrics_log["v1"].append({"err": v1_err, "iou": v1_iou, "ms": v1_time_ms, "status": v1_status})
        metrics_log["v2"].append({"err": v2_err, "iou": v2_iou, "ms": v2_time_ms, "status": v2_status})
        metrics_log["v3"].append({"err": v3_err, "iou": v3_iou, "ms": v3_time_ms, "status": v3_status})

        # Render On-Screen Display HUD Legend
        cv2.putText(display_frame, f"V1 (CSRT): {v1_status} Err:{v1_err:.1f}px", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display_frame, f"V2 (CSRT+Kalman): {v2_status} Err:{v2_err:.1f}px", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_frame, f"V3 (SiamRPN+Kalman): {v3_status} Err:{v3_err:.1f}px", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
    # COMPUTE OVERALL SUMMARY STATS
    # ----------------------------------------------------
    summary = {}
    for ver in ["v1", "v2", "v3"]:
        errs = [m["err"] for m in metrics_log[ver] if m["err"] < 900.0]
        ious = [m["iou"] for m in metrics_log[ver]]
        times = [m["ms"] for m in metrics_log[ver]]
        successes = [1 for m in metrics_log[ver] if (m["iou"] >= 0.4 or m["err"] <= 35.0)]

        summary[ver] = {
            "mean_error_px": round(float(np.mean(errs)), 2) if errs else 999.0,
            "median_error_px": round(float(np.median(errs)), 2) if errs else 999.0,
            "mean_iou": round(float(np.mean(ious)), 3),
            "success_rate_percent": round(len(successes) / len(metrics_log[ver]) * 100.0, 1),
            "mean_latency_ms": round(float(np.mean(times)), 2),
            "fps": round(1000.0 / float(np.mean(times)), 1) if np.mean(times) > 0 else 0,
            "lost_frames": sum(1 for m in metrics_log[ver] if m["status"] == "LOST")
        }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================================")
    print(" BENCHMARK SUMMARY RESULTS")
    print("==========================================================")
    for ver, s in summary.items():
        name = "V1 (CSRT)" if ver == "v1" else ("V2 (CSRT + Kalman)" if ver == "v2" else "V3 (SiamRPN + Kalman)")
        print(f"{name:25s} | Success: {s['success_rate_percent']:5.1f}% | Avg Err: {s['mean_error_px']:5.1f}px | IoU: {s['mean_iou']:.3f} | Speed: {s['fps']:4.1f} FPS ({s['mean_latency_ms']:4.1f}ms)")
    print("==========================================================")
    print(f"[BENCHMARK] Detailed CSV saved to '{CSV_PATH}'")
    print(f"[BENCHMARK] Summary JSON saved to '{SUMMARY_PATH}'\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--save-video", action="store_true", help="Save output visualization to MP4 file")
    args = parser.parse_args()

    run_benchmark(headless=args.headless, save_video=args.save_video)
