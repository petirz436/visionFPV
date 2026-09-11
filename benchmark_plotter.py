#!/usr/bin/env python3
"""
Statistical Plotter for FPV Tracker Benchmark Results
Reads 'benchmark_results.csv' & 'benchmark_summary.json' and generates 'benchmark_summary.png'
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

CSV_PATH = "benchmark_results.csv"
SUMMARY_PATH = "benchmark_summary.json"
OUTPUT_PLOT = "benchmark_summary.png"


def generate_benchmark_plots():
    if not os.path.exists(CSV_PATH) or not os.path.exists(SUMMARY_PATH):
        print("[ERROR] Benchmark data files missing. Please run 'benchmark_suite.py' first.")
        return

    print("[PLOTTER] Generating statistical visualization plots...")

    df = pd.read_csv(CSV_PATH)
    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    # Set dark modern plot style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("FPV Target Tracker Performance Benchmark (V1 vs V2 vs V3)", fontsize=16, fontweight='bold', y=0.98)

    frames = df["frame"]

    # Colors
    c_v1 = '#FF4B4B'  # Red
    c_v2 = '#2ECC71'  # Green
    c_v3 = '#00D2FF'  # Cyan

    # ----------------------------------------------------
    # Subplot 1: Center Distance Error (px)
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(frames, df["v1_err"].clip(upper=150), label="V1: CSRT", color=c_v1, linewidth=1.8)
    ax1.plot(frames, df["v2_err"].clip(upper=150), label="V2: CSRT + Kalman", color=c_v2, linewidth=1.8)
    ax1.plot(frames, df["v3_err"].clip(upper=150), label="V3: SiamRPN + Kalman", color=c_v3, linewidth=2.2)

    # Highlight Occlusion Zone (Frames 125..175)
    ax1.axvspan(125, 175, color='#E74C3C', alpha=0.25, label="Occlusion Zone")
    ax1.set_title("1. Center Position Distance Error (Pixels - Lower is Better)", fontsize=12)
    ax1.set_xlabel("Frame Number")
    ax1.set_ylabel("Error (px)")
    ax1.set_ylim(0, 150)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc="upper right")

    # ----------------------------------------------------
    # Subplot 2: IoU Overlap Accuracy (0.0 to 1.0)
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(frames, df["v1_iou"], label="V1: CSRT", color=c_v1, linewidth=1.8)
    ax2.plot(frames, df["v2_iou"], label="V2: CSRT + Kalman", color=c_v2, linewidth=1.8)
    ax2.plot(frames, df["v3_iou"], label="V3: SiamRPN + Kalman", color=c_v3, linewidth=2.2)

    ax2.axvspan(125, 175, color='#E74C3C', alpha=0.25)
    ax2.set_title("2. Intersection over Union (IoU Overlap - Higher is Better)", fontsize=12)
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("IoU Score")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc="upper right")

    # ----------------------------------------------------
    # Subplot 3: Processing Latency (ms) & FPS
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    trackers = ["V1 (CSRT)", "V2 (CSRT+Kalman)", "V3 (SiamRPN+Kalman)"]
    latencies = [summary["v1"]["mean_latency_ms"], summary["v2"]["mean_latency_ms"], summary["v3"]["mean_latency_ms"]]
    fps_vals = [summary["v1"]["fps"], summary["v2"]["fps"], summary["v3"]["fps"]]

    bars = ax3.bar(trackers, latencies, color=[c_v1, c_v2, c_v3], alpha=0.85, width=0.5)
    ax3.set_title("3. Average Frame Processing Latency (ms)", fontsize=12)
    ax3.set_ylabel("Latency (ms)")
    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, fps, lat in zip(bars, fps_vals, latencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f"{lat:.1f} ms\n({fps:.1f} FPS)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')

    # ----------------------------------------------------
    # Subplot 4: Overall Lock Success Rate (%)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    success_rates = [summary["v1"]["success_rate_percent"], summary["v2"]["success_rate_percent"], summary["v3"]["success_rate_percent"]]

    bars4 = ax4.bar(trackers, success_rates, color=[c_v1, c_v2, c_v3], alpha=0.85, width=0.5)
    ax4.set_title("4. Overall Lock Success Rate (%) Across All Test Phases", fontsize=12)
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_ylim(0, 115)
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, rate in zip(bars4, success_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, height + 2.0, f"{rate:.1f}%",
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    print(f"[PLOTTER] Benchmark statistical chart saved successfully to '{OUTPUT_PLOT}'.")
    return OUTPUT_PLOT


if __name__ == "__main__":
    generate_benchmark_plots()
