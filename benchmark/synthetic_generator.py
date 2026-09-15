#!/usr/bin/env python3
"""
Synthetic FPV Video & Ground-Truth Generator for Target Tracking Benchmark
Simulates realistic FPV drone scenarios with user-controlled parameters:
 1. Custom Ground Truth starting position & trajectory motion pattern
 2. Custom Obstacle Occlusion % (0% to 100% pillar width)
 3. Custom FPV Camera Vibration / Shake
 4. Custom Motion Blur & Jitter
"""

import cv2
import numpy as np
import json
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 480
FPS = 30
NUM_FRAMES = 300
OUTPUT_VIDEO = os.path.join(DATA_DIR, "synthetic_fpv_test.mp4")
OUTPUT_GT = os.path.join(DATA_DIR, "synthetic_gt.json")


def create_background(frame_idx, occlusion_pct=100.0, shake_offset=(0, 0)):
    """Generate dynamic background grid and obstacle based on occlusion_pct."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Grid lines with camera shake shift
    dx, dy = shake_offset
    grid_size = 40
    for x in range(-grid_size + dx, WIDTH + grid_size, grid_size):
        cv2.line(img, (x, 0), (x, HEIGHT), (30, 30, 35), 1)
    for y in range(-grid_size + dy, HEIGHT + grid_size, grid_size):
        cv2.line(img, (0, y), (WIDTH, y), (30, 30, 35), 1)

    # Obstacle pillar dimensions scaled by occlusion_pct
    base_w = 100.0 * (occlusion_pct / 100.0)
    center_x = 320 + dx
    x1 = int(center_x - base_w / 2.0)
    x2 = int(center_x + base_w / 2.0)
    y1 = 80 + dy
    y2 = 400 + dy

    if base_w > 5:
        cv2.rectangle(img, (x1, y1), (x2, y2), (45, 45, 50), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 100), 2)
        if base_w > 40:
            cv2.putText(img, "OBSTACLE", (x1 + 10, 240 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1)

    return img, (x1, y1, x2 - x1, y2 - y1)


def get_custom_target_position(frame_idx, num_frames=300, motion_pattern="zigzag",
                               start_x=80, start_y=150, occlusion_pct=100.0,
                               blur_level="light", obstacle_rect=(270, 80, 100, 320)):
    t = frame_idx
    w, h = 60, 60
    progress = t / float(max(1, num_frames))

    if motion_pattern == "smooth":
        cx = start_x + progress * (WIDTH - start_x - 100)
        cy = start_y + np.sin(progress * np.pi * 3) * 40.0
        phase = "Smooth Curved"
    elif motion_pattern == "circular":
        angle = progress * np.pi * 4
        radius = 80.0 + np.sin(progress * np.pi) * 40.0
        cx = start_x + 180 + np.cos(angle) * radius
        cy = start_y + np.sin(angle) * radius
        phase = "Circular Orbit"
    elif motion_pattern == "loop":
        cx = start_x + (np.sin(progress * np.pi * 4) * 0.5 + 0.5) * (WIDTH - start_x - 120)
        cy = start_y + np.cos(progress * np.pi * 4) * 60.0
        w = int(60 + np.sin(progress * np.pi * 6) * 15)
        h = w
        phase = "Looping & Scale"
    else:  # 'zigzag' default
        cx = start_x + progress * (WIDTH - start_x - 100)
        cy = start_y + np.sin(progress * np.pi * 6) * 75.0
        phase = "Agile ZigZag"

    x = int(cx - w / 2.0)
    y = int(cy - h / 2.0)

    # Determine occlusion
    obs_x, obs_y, obs_w, obs_h = obstacle_rect
    occluded = False
    if obs_w > 10 and (obs_x <= cx <= obs_x + obs_w) and (obs_y <= cy <= obs_y + obs_h):
        if occlusion_pct > 20.0:
            occluded = True

    # Determine blur
    blurred = False
    if blur_level == "heavy":
        blurred = (t % 3 != 0)
    elif blur_level == "light":
        blurred = (t % 5 == 0)

    return x, y, w, h, occluded, blurred, phase


def draw_target(img, bbox):
    """Draw synthetic target (orange/cyan/red crosshair pattern)."""
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), -1)  # Orange base
    cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 255, 0), -1)  # Cyan center
    cv2.circle(img, (x + w // 2, y + h // 2), max(2, w // 4), (0, 0, 255), -1)  # Red core dot


def generate_custom_synthetic_dataset(
    num_frames=300,
    motion_pattern="zigzag",
    occlusion_pct=100.0,
    shake_level="medium",
    blur_level="light",
    start_x=80,
    start_y=150,
    output_video=OUTPUT_VIDEO,
    output_gt=OUTPUT_GT
):
    print(f"[SYNTHETIC GENERATOR] Generating custom sequence: pattern={motion_pattern}, occlusion={occlusion_pct}%, shake={shake_level}, blur={blur_level}...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, FPS, (WIDTH, HEIGHT))

    gt_data = []
    shake_scale = 0.0
    if shake_level == "low":
        shake_scale = 2.0
    elif shake_level == "medium":
        shake_scale = 5.0
    elif shake_level == "high":
        shake_scale = 12.0

    for frame_idx in range(num_frames):
        # Calculate camera shake jitter
        dx = int(np.random.normal(0, shake_scale)) if shake_scale > 0 else 0
        dy = int(np.random.normal(0, shake_scale)) if shake_scale > 0 else 0

        frame, obs_rect = create_background(frame_idx, occlusion_pct=occlusion_pct, shake_offset=(dx, dy))
        x, y, w, h, occluded, blurred, phase = get_custom_target_position(
            frame_idx, num_frames=num_frames, motion_pattern=motion_pattern,
            start_x=start_x, start_y=start_y, occlusion_pct=occlusion_pct,
            blur_level=blur_level, obstacle_rect=obs_rect
        )

        # Apply camera shake to target as well
        x += dx
        y += dy

        if not occluded:
            draw_target(frame, (x, y, w, h))
        else:
            draw_target(frame, (x, y, w, h))
            obs_x, obs_y, obs_w, obs_h = obs_rect
            if obs_w > 5:
                cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (45, 45, 50), -1)
                cv2.rectangle(frame, (obs_x, obs_y), (obs_x + obs_w, obs_y + obs_h), (90, 90, 100), 2)
                if obs_w > 40:
                    cv2.putText(frame, "OBSTACLE", (obs_x + 10, 240 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 140), 1)

        if blurred:
            k_size = 11 if blur_level == "heavy" else 7
            kernel = np.zeros((k_size, k_size))
            kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
            kernel /= k_size
            frame = cv2.filter2D(frame, -1, kernel)

        cv2.putText(frame, f"PATTERN: {motion_pattern.upper()} ({phase})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"FRAME: {frame_idx + 1}/{num_frames}", (WIDTH - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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

    with open(output_gt, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"[SYNTHETIC GENERATOR] Saved video to '{output_video}' and Ground Truth to '{output_gt}'.")
    return output_video, output_gt


def generate_synthetic_dataset():
    """Default fallback generator."""
    return generate_custom_synthetic_dataset(
        num_frames=NUM_FRAMES, motion_pattern="zigzag", occlusion_pct=100.0,
        shake_level="medium", blur_level="light", start_x=80, start_y=150,
        output_video=OUTPUT_VIDEO, output_gt=OUTPUT_GT
    )


if __name__ == "__main__":
    generate_synthetic_dataset()
