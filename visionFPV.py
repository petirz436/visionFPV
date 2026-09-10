#0.1.0
#Discriminative Correlation Filter with Channel and Spatial Reliability


import cv2
import numpy as np
import time


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 0

# Ukuran default ROI kalau user hanya klik sekali
DEFAULT_ROI_W = 100
DEFAULT_ROI_H = 100

# Smoothing target center
ALPHA = 0.35

# Deadzone untuk error pixel
DEADZONE = 20

# Warna tidak perlu diubah; OpenCV default digunakan


# ============================================================
# TRACKER FACTORY
# ============================================================

def create_csrt_tracker():
    """
    Membuat CSRT tracker.
    Kompatibel dengan beberapa versi OpenCV.
    """

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tidak tersedia.\n"
        "Install opencv-contrib-python."
    )


# ============================================================
# GLOBAL STATE
# ============================================================

selecting = False
dragging = False

start_x = 0
start_y = 0

current_x = 0
current_y = 0

selected_bbox = None

tracker = None
tracking = False

smooth_cx = None
smooth_cy = None

last_time = time.time()


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global selecting
    global dragging

    global start_x
    global start_y

    global current_x
    global current_y

    global selected_bbox

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

        # ----------------------------------------------------
        # Jika hanya klik tanpa drag
        # buat ROI ukuran default di sekitar klik
        # ----------------------------------------------------

        if w < 10 or h < 10:

            x1 = x - DEFAULT_ROI_W // 2
            y1 = y - DEFAULT_ROI_H // 2

            w = DEFAULT_ROI_W
            h = DEFAULT_ROI_H

        selected_bbox = (x1, y1, w, h)


# ============================================================
# ROI VALIDATION
# ============================================================

def clamp_bbox(bbox, frame_width, frame_height):

    x, y, w, h = bbox

    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))

    w = max(10, min(w, frame_width - x))
    h = max(10, min(h, frame_height - y))

    return (x, y, w, h)


# ============================================================
# CALCULATE ERROR
# ============================================================

def calculate_error(target_cx, target_cy, frame_width, frame_height):

    frame_cx = frame_width / 2.0
    frame_cy = frame_height / 2.0

    error_x = target_cx - frame_cx
    error_y = target_cy - frame_cy

    return error_x, error_y


# ============================================================
# DEADZONE
# ============================================================

def apply_deadzone(value, deadzone):

    if abs(value) < deadzone:
        return 0.0

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    global selected_bbox
    global tracker
    global tracking

    global smooth_cx
    global smooth_cy

    global last_time

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        print("[ERROR] Kamera tidak dapat dibuka.")

        return

    # --------------------------------------------------------
    # WINDOW
    # --------------------------------------------------------

    window_name = "OpenCV Drone Target Tracker"

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print("==============================================")
    print(" OpenCV CSRT TARGET TRACKER")
    print("==============================================")
    print()
    print("[MOUSE]")
    print("  Drag  : pilih area target")
    print("  Click : pilih titik target")
    print()
    print("[KEYBOARD]")
    print("  R     : reset tracker")
    print("  ESC   : keluar")
    print()
    print("==============================================")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:

            print("[ERROR] Frame kamera gagal dibaca.")

            break

        frame_height, frame_width = frame.shape[:2]

        # ====================================================
        # CAMERA CENTER
        # ====================================================

        frame_cx = frame_width // 2
        frame_cy = frame_height // 2

        # crosshair kamera
        cv2.drawMarker(
            frame,
            (frame_cx, frame_cy),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )

        # ====================================================
        # USER SEDANG MEMILIH ROI
        # ====================================================

        if selecting:

            x1 = min(start_x, current_x)
            y1 = min(start_y, current_y)

            x2 = max(start_x, current_x)
            y2 = max(start_y, current_y)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

        # ====================================================
        # ADA ROI BARU
        # ====================================================

        if selected_bbox is not None:

            bbox = clamp_bbox(
                selected_bbox,
                frame_width,
                frame_height
            )

            try:

                tracker = create_csrt_tracker()

                tracker.init(
                    frame,
                    bbox
                )

                tracking = True

                x, y, w, h = bbox

                smooth_cx = x + w / 2.0
                smooth_cy = y + h / 2.0

                print(
                    "[TRACKER] Target initialized:",
                    bbox
                )

            except Exception as e:

                print(
                    "[ERROR] Gagal initialize tracker:",
                    e
                )

                tracking = False

            selected_bbox = None

        # ====================================================
        # TRACK
        # ====================================================

        if tracking and tracker is not None:

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = bbox

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                # --------------------------------------------
                # VALIDATION
                # --------------------------------------------

                valid = (
                    w > 5 and
                    h > 5 and
                    x + w > 0 and
                    y + h > 0 and
                    x < frame_width and
                    y < frame_height
                )

                if valid:

                    # ----------------------------------------
                    # TARGET CENTER
                    # ----------------------------------------

                    target_cx = x + w / 2.0
                    target_cy = y + h / 2.0

                    # ----------------------------------------
                    # EMA SMOOTHING
                    # ----------------------------------------

                    if smooth_cx is None:
                        smooth_cx = target_cx

                    if smooth_cy is None:
                        smooth_cy = target_cy

                    smooth_cx = (
                        ALPHA * target_cx
                        +
                        (1.0 - ALPHA) * smooth_cx
                    )

                    smooth_cy = (
                        ALPHA * target_cy
                        +
                        (1.0 - ALPHA) * smooth_cy
                    )

                    # ----------------------------------------
                    # ERROR
                    # ----------------------------------------

                    error_x, error_y = calculate_error(
                        smooth_cx,
                        smooth_cy,
                        frame_width,
                        frame_height
                    )

                    # deadzone
                    error_x_control = apply_deadzone(
                        error_x,
                        DEADZONE
                    )

                    error_y_control = apply_deadzone(
                        error_y,
                        DEADZONE
                    )

                    # ----------------------------------------
                    # DRAW TARGET
                    # ----------------------------------------

                    x2 = x + w
                    y2 = y + h

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # target center
                    target_center = (
                        int(smooth_cx),
                        int(smooth_cy)
                    )

                    cv2.circle(
                        frame,
                        target_center,
                        6,
                        (0, 0, 255),
                        -1
                    )

                    # line target -> camera center
                    cv2.line(
                        frame,
                        (frame_cx, frame_cy),
                        target_center,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------------
                    # STATUS
                    # ----------------------------------------

                    cv2.putText(
                        frame,
                        "TRACKING",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"CX: {smooth_cx:.1f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"CY: {smooth_cy:.1f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ERR X: {error_x_control:.1f}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"ERR Y: {error_y_control:.1f}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------------
                    # NORMALIZED ERROR
                    # -1 ... +1
                    # ----------------------------------------

                    normalized_x = error_x / (frame_width / 2.0)
                    normalized_y = error_y / (frame_height / 2.0)

                    normalized_x = np.clip(
                        normalized_x,
                        -1.0,
                        1.0
                    )

                    normalized_y = np.clip(
                        normalized_y,
                        -1.0,
                        1.0
                    )

                    cv2.putText(
                        frame,
                        f"NORM X: {normalized_x:.2f}",
                        (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"NORM Y: {normalized_y:.2f}",
                        (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------------
                    # DEADZONE VISUAL
                    # ----------------------------------------

                    deadzone_x = int(DEADZONE)
                    deadzone_y = int(DEADZONE)

                    cv2.rectangle(
                        frame,
                        (
                            frame_cx - deadzone_x,
                            frame_cy - deadzone_y
                        ),
                        (
                            frame_cx + deadzone_x,
                            frame_cy + deadzone_y
                        ),
                        (255, 255, 255),
                        1
                    )

                    # ----------------------------------------
                    # PRINT DATA
                    # ----------------------------------------

                    now = time.time()
                    dt = now - last_time
                    last_time = now

                    fps = 1.0 / dt if dt > 0 else 0

                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (frame_width - 150, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    print(
                        f"\r"
                        f"target=({smooth_cx:7.1f}, {smooth_cy:7.1f}) "
                        f"error=({error_x:7.1f}, {error_y:7.1f}) "
                        f"norm=({normalized_x:+.2f}, {normalized_y:+.2f})",
                        end=""
                    )

                else:

                    tracking = False

            else:

                # ===========================================
                # TARGET LOST
                # ===========================================

                cv2.putText(
                    frame,
                    "TARGET LOST",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "CLICK / DRAG TARGET",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            window_name,
            frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            # ESC
            break

        elif key == ord("r"):

            print("\n[TRACKER] Reset")

            tracker = None
            tracking = False

            smooth_cx = None
            smooth_cy = None

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()