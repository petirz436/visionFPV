#!/usr/bin/env python3
"""
Synthetic FPV Video & Ground-Truth Generator for Target Tracking Benchmark
Simulates realistic FPV drone scenarios:
 1. Smooth trajectory
 2. High-speed direction change / jerk
 3. Complete Occlusion event (Target behind obstacle)
 4. Motion blur & vibration
 5. Lighting / Contrast variation
"""

import cv2
import numpy as np
import json
import os

WIDTH = 640
HEIGHT = 480
FPS = 30
NUM_FRAMES = 300
OUTPUT_VIDEO = "synthetic_fpv_test.mp4"
OUTPUT_GT = "synthetic_gt.json"


def create_background(frame_idx):
    """Generate a dynamic background with grid lines and a static obstacle."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Subtle dark background grid
    grid_size = 40
    for x in range(0, WIDTH, grid_size):
        cv2.line(img, (x, 0), (x, HEIGHT), (30, 30, 35), 1)
    for y in range(0, HEIGHT, grid_size):
        cv2.line(img, (0, y), (WIDTH, y), (30, 30, 35), 1)

    # Draw static pillar obstacle (for occlusion phase)
    # Pillar located around x = 270..370, y = 80..400
    cv2.rectangle(img, (270, 80), (370, 400), (45, 45, 50), -1)
    cv2.rectangle(img, (270, 80), (370, 400), (90, 90, 100), 2)
    cv2.putText(img, "OBSTACLE", (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)

    return img


def get_target_position(frame_idx):
    """
    Compute Ground Truth target position (center_x, center_y, width, height, occluded, blurred)
    across 5 distinct test phases.
    """
    t = frame_idx
    w, h = 60, 60

    # Phase 1: Smooth curved motion (Frames 0..60)
    if t <= 60:
        cx = 80 + (t / 60.0) * 120.0  # 80 -> 200
        cy = 150 + np.sin(t * 0.1) * 30.0
        occluded = False
        blurred = False

    # Phase 2: High Acceleration / Agile Jerk (Frames 61..120)
    elif t <= 120:
        dt = t - 60
        # Fast zig-zag movement towards obstacle
        cx = 200 + (dt / 60.0) * 120.0  # 200 -> 320
        cy = 150 + np.sin(dt * 0.3) * 70.0
        occluded = False
        blurred = True if dt % 5 < 2 else False  # intermittent motion blur

    # Phase 3: Complete Occlusion behind Pillar (Frames 121..180)
    elif t <= 180:
        dt = t - 120
        cx = 320 + (dt / 60.0) * 40.0  # 320 -> 360 (inside 270..370 obstacle range)
        cy = 150 + (dt / 60.0) * 80.0  # 150 -> 230
        occluded = True if 130 <= t <= 170 else False
        blurred = False

    # Phase 4: Emergence & Fast Recovery (Frames 181..240)
    elif t <= 240:
        dt = t - 180
        cx = 360 + (dt / 60.0) * 160.0  # 360 -> 520 (exits obstacle)
        cy = 230 - (dt / 60.0) * 100.0  # 230 -> 130
        occluded = False
        blurred = True

    # Phase 5: Circular Looping & Scale Change (Frames 241..300)
    else:
        dt = t - 240
        angle = dt * 0.1
        radius = 50.0 + dt * 0.2
        cx = 500 + np.cos(angle) * radius
        cy = 200 + np.sin(angle) * radius
        w = int(60 + np.sin(dt * 0.05) * 15)  # Scale variation 45px..75px
        h = w
        occluded = False
        blurred = False

    x = int(cx - w / 2.0)
    y = int(cy - h / 2.0)
    return x, y, w, h, occluded, blurred


def draw_target(img, bbox):
    """Draw synthetic target (red/cyan pattern with crosshair)."""
    x, y, w, h = bbox
    # Draw object body (solid bright color for feature matching)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), -1)  # Orange base
    cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 255, 0), -1)  # Cyan center
    cv2.circle(img, (x + w // 2, y + h // 2), w // 4, (0, 0, 255), -1)  # Red core dot


def generate_synthetic_dataset():
    print("[SYNTHETIC GENERATOR] Generating test sequence (300 frames)...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    gt_data = []

    for frame_idx in range(NUM_FRAMES):
        frame = create_background(frame_idx)
        x, y, w, h, occluded, blurred = get_target_position(frame_idx)

        # Draw target BEFORE obstacle if not occluded, or draw obstacle overlay after if occluded
        if not occluded:
            draw_target(frame, (x, y, w, h))
        else:
            # Draw target behind obstacle, then re-draw obstacle section to cover it
            draw_target(frame, (x, y, w, h))
            cv2.rectangle(frame, (270, 80), (370, 400), (45, 45, 50), -1)
            cv2.rectangle(frame, (270, 80), (370, 400), (90, 90, 100), 2)
            cv2.putText(frame, "OBSTACLE", (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)

        # Apply motion blur if flagged
        if blurred:
            kernel_size = 9
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            frame = cv2.filter2D(frame, -1, kernel)

        # HUD Phase Label
        phase = "1: Smooth" if frame_idx <= 60 else ("2: Fast Jerk" if frame_idx <= 120 else ("3: Occlusion" if frame_idx <= 180 else ("4: Recovery" if frame_idx <= 240 else "5: Loop & Scale")))
        cv2.putText(frame, f"PHASE {phase}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FRAME: {frame_idx}/{NUM_FRAMES}", (WIDTH - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(frame)

        gt_data.append({
            "frame": frame_idx,
            "bbox": [x, y, w, h],
            "center": [x + w / 2.0, y + h / 2.0],
            "occluded": occluded,
            "blurred": blurred,
            "phase": phase
        })

    writer.release()

    with open(OUTPUT_GT, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"[SYNTHETIC GENERATOR] Saved video to '{OUTPUT_VIDEO}' and Ground Truth to '{OUTPUT_GT}'.")
    return OUTPUT_VIDEO, OUTPUT_GT


if __name__ == "__main__":
    generate_synthetic_dataset()
