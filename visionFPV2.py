# OpenCV CSRT Target Tracker with Kalman Filter Motion Prediction & Pre-processing Pipeline
# Version: 0.2.0 (FPV Motion Resilience Upgrade)

import cv2
import numpy as np
import time

# ============================================================
# CONFIGURATION & HYPERPARAMETERS
# ============================================================

CAMERA_INDEX = 0

# Default ROI box size when single-clicked
DEFAULT_ROI_W = 100
DEFAULT_ROI_H = 100

# Exponential Moving Average (EMA) smoothing for target center output
ALPHA = 0.35

# Deadzone threshold for drone orientation error (pixels)
DEADZONE = 20

# Maximum consecutive frames to rely on Kalman Filter prediction during lost lock / motion blur
MAX_LOST_FRAMES = 5

# Pre-processing mode: 'clahe', 'equalize', or 'none'
DEFAULT_PREPROCESS_MODE = 'clahe'


# ============================================================
# KALMAN FILTER CLASS (2D CONSTANT VELOCITY MODEL)
# ============================================================

class Kalman2DTracker:
    """
    2D Kalman Filter for motion prediction and state estimation.
    State Vector    : [x, y, vx, vy]^T
    Measurement Vector: [zx, zy]^T
    """
    def __init__(self, process_noise_std=1e-2, measurement_noise_std=1e-1):
        # 4 dynamic params (x, y, vx, vy), 2 measurement params (zx, zy)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # State transition matrix F (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance matrix Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise_std
        # Allow velocity to change rapidly for agile FPV drone maneuvers
        self.kf.processNoiseCov[2, 2] *= 5.0
        self.kf.processNoiseCov[3, 3] *= 5.0

        # Measurement noise covariance matrix R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise_std

        # Error covariance matrix P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False

    def init(self, cx, cy):
        """Initialize filter state at target coordinate (cx, cy)."""
        self.kf.statePost = np.array([[np.float32(cx)],
                                      [np.float32(cy)],
                                      [0.0],
                                      [0.0]], dtype=np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = True

    def predict(self, dt=1.0):
        """Predict the next state based on current velocity and dt."""
        if not self.initialized:
            return 0.0, 0.0, 0.0, 0.0

        # Update dt in transition matrix
        self.kf.transitionMatrix[0, 2] = np.float32(dt)
        self.kf.transitionMatrix[1, 3] = np.float32(dt)

        prediction = self.kf.predict()
        pred_cx = float(prediction[0, 0])
        pred_cy = float(prediction[1, 0])
        vx = float(prediction[2, 0])
        vy = float(prediction[3, 0])

        return pred_cx, pred_cy, vx, vy

    def correct(self, cx, cy):
        """Correct the state estimation with observed measurement (cx, cy)."""
        if not self.initialized:
            self.init(cx, cy)
            return cx, cy, 0.0, 0.0

        measurement = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
        estimated = self.kf.correct(measurement)

        est_cx = float(estimated[0, 0])
        est_cy = float(estimated[1, 0])
        vx = float(estimated[2, 0])
        vy = float(estimated[3, 0])

        return est_cx, est_cy, vx, vy


# ============================================================
# IMAGE PRE-PROCESSING PIPELINE
# ============================================================

def preprocess_frame(frame, mode='clahe'):
    """
    Applies image pre-processing pipeline for illumination robustness.
    Modes:
      - 'clahe': Contrast Limited Adaptive Histogram Equalization (Grayscale -> CLAHE -> 3-channel BGR)
      - 'equalize': Standard Histogram Equalization (Grayscale -> Equalize -> 3-channel BGR)
      - 'none': Raw BGR frame
    """
    if mode == 'none':
        return frame.copy()

    # Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
    elif mode == 'equalize':
        enhanced = cv2.equalizeHist(gray)
    else:
        enhanced = gray

    # Convert back to 3-channel image for full OpenCV tracker compatibility
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ============================================================
# TRACKER FACTORY
# ============================================================

def create_csrt_tracker():
    """Membuat instance CSRT tracker kompatibel dengan berbagai versi OpenCV."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT Tracker tidak ditemukan. Install opencv-contrib-python.")


# ============================================================
# GLOBAL STATE & MOUSE HANDLER
# ============================================================

selecting = False
dragging = False
start_x = 0
start_y = 0
current_x = 0
current_y = 0
selected_bbox = None


def mouse_callback(event, x, y, flags, param):
    global selecting, dragging, start_x, start_y, current_x, current_y, selected_bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        dragging = False
        start_x = x
        start_y = y
        current_x = x
        current_y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            current_x = x
            current_y = y
            if abs(current_x - start_x) > 5 or abs(current_y - start_y) > 5:
                dragging = True

    elif event == cv2.EVENT_LBUTTONUP:
        if not selecting:
            return
        selecting = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        w = x2 - x1
        h = y2 - y1

        if w < 10 or h < 10:
            x1 = x - DEFAULT_ROI_W // 2
            y1 = y - DEFAULT_ROI_H // 2
            w = DEFAULT_ROI_W
            h = DEFAULT_ROI_H

        selected_bbox = (x1, y1, w, h)


def clamp_bbox(bbox, frame_width, frame_height):
    x, y, w, h = bbox
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(10, min(w, frame_width - x))
    h = max(10, min(h, frame_height - y))
    return (x, y, w, h)


def calculate_error(target_cx, target_cy, frame_width, frame_height):
    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0
    return target_cx - frame_cx, target_cy - frame_cy


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    return value


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

def main():
    global selected_bbox

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka.")
        return

    window_name = "FPV Target Tracker + Kalman Prediction & Pipeline"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # State variables
    tracker = None
    tracking = False
    kalman = Kalman2DTracker()
    preprocess_mode = DEFAULT_PREPROCESS_MODE
    
    lost_counter = 0
    is_predicting = False
    
    smooth_cx = None
    smooth_cy = None
    last_bbox_wh = (DEFAULT_ROI_W, DEFAULT_ROI_H)
    target_velocity = (0.0, 0.0)

    last_time = time.time()

    print()
    print("==========================================================")
    print(" FPV DRONE TARGET TRACKER v0.2.0")
    print(" Algoritma: CSRT + Kalman Filter Motion Prediction")
    print(" Pipeline : Grayscale + Histogram Equalization / CLAHE")
    print("==========================================================")
    print()
    print("[MOUSE]")
    print("  Drag  : Pilih ROI target")
    print("  Click : Pilih titik target (Default ROI)")
    print()
    print("[KEYBOARD]")
    print("  P     : Toggle Pre-processing Mode (CLAHE -> EQUALIZE -> OFF)")
    print("  R     : Reset Tracker")
    print("  ESC   : Keluar")
    print("==========================================================")
    print()

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[ERROR] Frame kamera gagal dibaca.")
            break

        frame_height, frame_width = raw_frame.shape[:2]
        frame_cx = frame_width // 2
        frame_cy = frame_height // 2

        # ----------------------------------------------------
        # 1. PRE-PROCESSING PIPELINE
        # ----------------------------------------------------
        processed_frame = preprocess_frame(raw_frame, mode=preprocess_mode)
        
        # Display base frame (raw for standard visualization with HUD)
        display_frame = raw_frame.copy()

        # Draw camera center crosshair
        cv2.drawMarker(
            display_frame,
            (frame_cx, frame_cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )

        # ----------------------------------------------------
        # 2. USER ROI SELECTION (DRAGGING)
        # ----------------------------------------------------
        if selecting:
            x1 = min(start_x, current_x)
            y1 = min(start_y, current_y)
            x2 = max(start_x, current_x)
            y2 = max(start_y, current_y)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # ----------------------------------------------------
        # 3. INITIALIZE TRACKER WITH NEW ROI
        # ----------------------------------------------------
        if selected_bbox is not None:
            bbox = clamp_bbox(selected_bbox, frame_width, frame_height)
            try:
                tracker = create_csrt_tracker()
                tracker.init(processed_frame, bbox)
                tracking = True

                x, y, w, h = bbox
                init_cx = x + w / 2.0
                init_cy = y + h / 2.0

                kalman.init(init_cx, init_cy)
                smooth_cx = init_cx
                smooth_cy = init_cy
                last_bbox_wh = (w, h)
                lost_counter = 0
                is_predicting = False

                print(f"[TRACKER] Target terkunci: ROI={bbox}, Center=({init_cx:.1f}, {init_cy:.1f})")
            except Exception as e:
                print(f"[ERROR] Gagal inisialisasi tracker: {e}")
                tracking = False

            selected_bbox = None

        # ----------------------------------------------------
        # 4. TRACKING & KALMAN MOTION PREDICTION
        # ----------------------------------------------------
        now = time.time()
        dt = max(now - last_time, 0.001)
        last_time = now

        target_cx, target_cy = None, None

        if tracking and tracker is not None:
            success, bbox = tracker.update(processed_frame)

            if success:
                x, y, w, h = bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                valid = (
                    w > 5 and h > 5 and
                    x + w > 0 and y + h > 0 and
                    x < frame_width and y < frame_height
                )

                if valid:
                    meas_cx = x + w / 2.0
                    meas_cy = y + h / 2.0
                    last_bbox_wh = (w, h)

                    # Update Kalman Filter with observation (Correct & Predict)
                    est_cx, est_cy, vx, vy = kalman.correct(meas_cx, meas_cy)
                    pred_cx, pred_cy, _, _ = kalman.predict(dt=1.0)

                    target_cx = est_cx
                    target_cy = est_cy
                    target_velocity = (vx, vy)
                    
                    lost_counter = 0
                    is_predicting = False

                    # Draw Bounding Box (Solid Green for Active CSRT Tracking)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    success = False

            # If CSRT tracking fails (motion blur, obstruction, lost lock)
            if not success:
                lost_counter += 1

                if lost_counter <= MAX_LOST_FRAMES:
                    # Rely on Kalman Filter prediction to maintain aim!
                    pred_cx, pred_cy, vx, vy = kalman.predict(dt=1.0)
                    target_cx = pred_cx
                    target_cy = pred_cy
                    target_velocity = (vx, vy)
                    is_predicting = True

                    # Predict bounding box position
                    w, h = last_bbox_wh
                    pred_x = int(pred_cx - w / 2.0)
                    pred_y = int(pred_cy - h / 2.0)

                    # Attempt CSRT tracker re-alignment at predicted ROI for upcoming frames
                    try:
                        pred_bbox = clamp_bbox((pred_x, pred_y, w, h), frame_width, frame_height)
                        tracker = create_csrt_tracker()
                        tracker.init(processed_frame, pred_bbox)
                    except Exception:
                        pass

                    # Draw Predicted Bounding Box (Orange for Kalman Prediction)
                    cv2.rectangle(
                        display_frame,
                        (pred_x, pred_y),
                        (pred_x + w, pred_y + h),
                        (0, 165, 255),
                        2
                    )
                else:
                    # Lost target exceeds max threshold
                    tracking = False
                    is_predicting = False

        # ----------------------------------------------------
        # 5. SMOOTHING & CONTROLLER ERROR COMPUTATION
        # ----------------------------------------------------
        if target_cx is not None and target_cy is not None:
            if smooth_cx is None:
                smooth_cx = target_cx
            if smooth_cy is None:
                smooth_cy = target_cy

            smooth_cx = ALPHA * target_cx + (1.0 - ALPHA) * smooth_cx
            smooth_cy = ALPHA * target_cy + (1.0 - ALPHA) * smooth_cy

            error_x, error_y = calculate_error(smooth_cx, smooth_cy, frame_width, frame_height)
            error_x_control = apply_deadzone(error_x, DEADZONE)
            error_y_control = apply_deadzone(error_y, DEADZONE)

            target_center = (int(smooth_cx), int(smooth_cy))

            # Draw target center point
            center_color = (0, 165, 255) if is_predicting else (0, 0, 255)
            cv2.circle(display_frame, target_center, 6, center_color, -1)

            # Draw trajectory vector line (Camera Center -> Target Center)
            cv2.line(display_frame, (frame_cx, frame_cy), target_center, (255, 255, 255), 2)

            # Draw Velocity Arrow (Motion vector predicted by Kalman)
            vx, vy = target_velocity
            if abs(vx) > 0.5 or abs(vy) > 0.5:
                vel_end = (int(smooth_cx + vx * 5.0), int(smooth_cy + vy * 5.0))
                cv2.arrowedLine(display_frame, target_center, vel_end, (0, 255, 255), 2, tipLength=0.3)

            # Telemetry text on HUD
            status_text = f"PREDICTING [KALMAN] ({lost_counter}/{MAX_LOST_FRAMES})" if is_predicting else "TRACKING (CSRT)"
            status_color = (0, 165, 255) if is_predicting else (0, 255, 0)
            
            cv2.putText(display_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_frame, f"CX: {smooth_cx:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"CY: {smooth_cy:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"ERR X: {error_x_control:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"ERR Y: {error_y_control:.1f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Normalized Error (-1.0 ... +1.0)
            norm_x = np.clip(error_x / (frame_width / 2.0), -1.0, 1.0)
            norm_y = np.clip(error_y / (frame_height / 2.0), -1.0, 1.0)

            cv2.putText(display_frame, f"NORM X: {norm_x:+.2f}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"NORM Y: {norm_y:+.2f}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"VEL: ({vx:+.1f}, {vy:+.1f})", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            print(
                f"\r[{'PRED' if is_predicting else 'TRK '}] target=({smooth_cx:6.1f}, {smooth_cy:6.1f}) "
                f"err=({error_x:6.1f}, {error_y:6.1f}) norm=({norm_x:+.2f}, {norm_y:+.2f}) vel=({vx:+.1f}, {vy:+.1f})",
                end=""
            )
        else:
            status_text = "TARGET LOST" if (tracking is False and lost_counter > MAX_LOST_FRAMES) else "CLICK / DRAG TARGET"
            status_color = (0, 0, 255) if status_text == "TARGET LOST" else (255, 255, 255)
            cv2.putText(display_frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # ----------------------------------------------------
        # 6. HUD OVERLAY (DEADZONE, PIPELINE MODE & FPS)
        # ----------------------------------------------------
        # Deadzone rectangle
        dz_x, dz_y = int(DEADZONE), int(DEADZONE)
        cv2.rectangle(display_frame, (frame_cx - dz_x, frame_cy - dz_y), (frame_cx + dz_x, frame_cy + dz_y), (255, 255, 255), 1)

        # Pipeline status
        cv2.putText(display_frame, f"PIPELINE: {preprocess_mode.upper()}", (frame_width - 240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 250, 200), 2)

        # FPS counter
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (frame_width - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Optional PIP (Picture in Picture) showing pre-processed feed at bottom-right
        pip_h, pip_w = 120, 160
        small_proc = cv2.resize(processed_frame, (pip_w, pip_h))
        display_frame[frame_height - pip_h - 10:frame_height - 10, frame_width - pip_w - 10:frame_width - 10] = small_proc
        cv2.rectangle(display_frame, (frame_width - pip_w - 10, frame_height - pip_h - 10), (frame_width - 10, frame_height - 10), (0, 255, 255), 1)
        cv2.putText(display_frame, "PREPROCESSED", (frame_width - pip_w - 5, frame_height - pip_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ----------------------------------------------------
        # 7. DISPLAY & KEYBOARD INPUT
        # ----------------------------------------------------
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("r") or key == ord("R"):
            print("\n[TRACKER] Tracker Reset.")
            tracker = None
            tracking = False
            lost_counter = 0
            is_predicting = False
            smooth_cx = None
            smooth_cy = None
        elif key == ord("p") or key == ord("P"):
            modes = ['clahe', 'equalize', 'none']
            preprocess_mode = modes[(modes.index(preprocess_mode) + 1) % len(modes)]
            print(f"\n[PIPELINE] Switched Pre-processing mode to: {preprocess_mode.upper()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
