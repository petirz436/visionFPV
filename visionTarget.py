import cv2 as cv
import numpy as np
import time
import urllib.request

# ============================================================
# CONFIGURATION
# ============================================================
USE_ESP_CAM = True                             # Set True untuk stream ESP32-CAM, False untuk Webcam lokal
ESP_CAM_URL = "http://172.20.10.3/stream" # URL stream ESP32-CAM (contoh: http://<IP>:81/stream atau http://<IP>/mjpeg)
CAMERA_INDEX = 0                               # Index webcam lokal jika USE_ESP_CAM = False
DEFAULT_ROI_W = 100
DEFAULT_ROI_H = 100
ALPHA = 0.35            # Factor smoothing EMA
DEADZONE = 20           # Pixel deadzone error
MIN_CONTOUR_AREA = 500  # Filter kontur minimal untuk deteksi target


# ============================================================
# ESP32-CAM STREAM HANDLER
# ============================================================
class ESPCamStream:
    """
    Class pembaca stream MJPEG HTTP dari ESP32-CAM.
    Kompatibel dengan antarmuka cv.VideoCapture (.read(), .isOpened(), .release()).
    """
    def __init__(self, url):
        self.url = url
        self.stream = None
        self.bytes = b''
        self._is_opened = False
        self.connect()

    def connect(self):
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            self.stream = urllib.request.urlopen(req, timeout=5)
            self._is_opened = True
            print(f"[ESP-CAM] Berhasil terhubung ke stream: {self.url}")
        except Exception as e:
            print(f"[ERROR] Gagal terhubung ke ESP32-CAM ({self.url}): {e}")
            self._is_opened = False

    def isOpened(self):
        return self._is_opened

    def read(self):
        if not self._is_opened or self.stream is None:
            return False, None

        try:
            while True:
                self.bytes += self.stream.read(1024)
                a = self.bytes.find(b'\xff\xd8')  # JPEG SOI
                b = self.bytes.find(b'\xff\xd9')  # JPEG EOI
                if a != -1 and b != -1:
                    jpg = self.bytes[a:b+2]
                    self.bytes = self.bytes[b+2:]
                    frame = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
        except Exception as e:
            print(f"[ESP-CAM] Stream error / frame dropped: {e}")
            self._is_opened = False
            return False, None

    def release(self):
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self._is_opened = False


def get_camera_capture(source):
    """
    Membuka capture video dari webcam lokal (int index) atau stream URL ESP32-CAM (str URL).
    """
    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://")):
        print(f"[INFO] Membuka stream ESP32-CAM: {source}")
        cap = cv.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            return cap
        else:
            print("[INFO] Fallback ke custom HTTP MJPEG Stream Reader...")
            return ESPCamStream(source)
    else:
        print(f"[INFO] Membuka webcam lokal index: {source}")
        return cv.VideoCapture(source)


# HSV Color Mask
# #biru
# HSV_LOWER = np.array([90, 100, 100])
# HSV_UPPER = np.array([130, 255, 255])
# #hijau
# HSV_LOWER = np.array([35, 100, 100])
# HSV_UPPER = np.array([85, 255, 255])
# #merah
# HSV_LOWER = np.array([0, 100, 100])
# HSV_UPPER = np.array([25, 255, 255])
#coklat muka gw
HSV_LOWER = np.array([0, 20, 50])
HSV_UPPER = np.array([20, 170, 255])

# ============================================================
# TRACKER FACTORY (dari visionFPV.py)
# ============================================================
def create_csrt_tracker():
    """Membuat CSRT Tracker compatible dengan berbagai versi OpenCV."""
    if hasattr(cv, "TrackerCSRT_create"):
        return cv.TrackerCSRT_create()
    if hasattr(cv, "legacy") and hasattr(cv.legacy, "TrackerCSRT_create"):
        return cv.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tidak tersedia. Install opencv-contrib-python.")


# ============================================================
# GLOBAL STATE & MOUSE CALLBACK
# ============================================================
selecting = False
start_x, start_y = 0, 0
current_x, current_y = 0, 0

selected_bbox = None
click_point = None


def mouse_callback(event, x, y, flags, param):
    global selecting, start_x, start_y, current_x, current_y, selected_bbox, click_point

    if event == cv.EVENT_LBUTTONDOWN:
        selecting = True
        start_x, start_y = x, y
        current_x, current_y = x, y
        click_point = (x, y)

    elif event == cv.EVENT_MOUSEMOVE:
        if selecting:
            current_x, current_y = x, y

    elif event == cv.EVENT_LBUTTONUP:
        if not selecting:
            return
        selecting = False

        x1 = min(start_x, x)
        y1 = min(start_y, y)
        x2 = max(start_x, x)
        y2 = max(start_y, y)

        w = x2 - x1
        h = y2 - y1

        # Jika drag lebih dari 10px, anggap manual ROI drag
        if w >= 10 and h >= 10:
            selected_bbox = (x1, y1, w, h)
            click_point = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def clamp_bbox(bbox, frame_width, frame_height):
    x, y, w, h = bbox
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(10, min(w, frame_width - x))
    h = max(10, min(h, frame_height - y))
    return (x, y, w, h)


def apply_deadzone(value, deadzone):
    if abs(value) < deadzone:
        return 0.0
    return value


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    global selected_bbox, click_point

    source = ESP_CAM_URL if USE_ESP_CAM else CAMERA_INDEX
    cap = get_camera_capture(source)
    if not cap.isOpened():
        print(f"[ERROR] Kamera / Stream ({source}) tidak dapat dibuka.")
        return

    window_name = "Multi-Target Vision Lock System"
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, mouse_callback)

    tracker = None
    tracking = False
    smooth_cx, smooth_cy = None, None
    last_time = time.time()

    print("\n==============================================")
    print(" MULTI-TARGET LOCK & VISION SYSTEM")
    print("==============================================")
    print(f"[STREAM SOURCE]: {'ESP32-CAM (' + ESP_CAM_URL + ')' if USE_ESP_CAM else f'Webcam Index {CAMERA_INDEX}'}")
    print("[CARA PILIH TARGET]")
    print(" 1. Klik pada kotak target yang terdeteksi")
    print(" 2. Atau tekan tombol Angka (1-9) sesuai nomor target")
    print(" 3. Atau Drag mouse untuk membuat ROI manual")
    print()
    print("[KONTROL KEYBOARD]")
    print(" R   : Reset lock / kembali ke deteksi multi-target")
    print(" ESC : Keluar")
    print("==============================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        frame_h, frame_w = frame.shape[:2]
        frame_cx, frame_cy = frame_w // 2, frame_h // 2

        # Crosshair kamera utama
        cv.drawMarker(
            frame,
            (frame_cx, frame_cy),
            (255, 255, 255),
            cv.MARKER_CROSS,
            20,
            2,
        )

        # ----------------------------------------------------
        # 1. MULTI-TARGET DETECTION (Gunakan HSV Mask dari visionTarget.py)
        # ----------------------------------------------------
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)

        contours, _ = cv.findContours(
            mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        detected_targets = []
        target_id_count = 1

        for c in contours:
            area = cv.contourArea(c)
            if area >= MIN_CONTOUR_AREA:
                x, y, w, h = cv.boundingRect(c)
                M = cv.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                detected_targets.append(
                    {
                        "id": target_id_count,
                        "bbox": (x, y, w, h),
                        "center": (cx, cy),
                        "area": area,
                    }
                )
                target_id_count += 1

        # ----------------------------------------------------
        # 2. SELEKSI TARGET DENGAN KLIK MOUSE
        # ----------------------------------------------------
        if click_point is not None and not tracking:
            px, py = click_point
            clicked_on_target = False

            # Cek apakah titik klik berada di dalam kotak salah satu target
            for t in detected_targets:
                tx, ty, tw, th = t["bbox"]
                if tx <= px <= tx + tw and ty <= py <= ty + th:
                    selected_bbox = t["bbox"]
                    clicked_on_target = True
                    print(f"[LOCK] Target #{t['id']} dipilih via Klik Mouse!")
                    break

            # Jika klik di area kosong, buat ROI manual ukuran default
            if not clicked_on_target:
                x1 = max(0, px - DEFAULT_ROI_W // 2)
                y1 = max(0, py - DEFAULT_ROI_H // 2)
                selected_bbox = (x1, y1, DEFAULT_ROI_W, DEFAULT_ROI_H)
                print(f"[LOCK] Custom ROI dibuat di ({px}, {py})")

            click_point = None

        # ----------------------------------------------------
        # 3. INITIALIZE TRACKER JIKA BBOX DIPILIH
        # ----------------------------------------------------
        if selected_bbox is not None:
            bbox = clamp_bbox(selected_bbox, frame_w, frame_h)
            try:
                tracker = create_csrt_tracker()
                tracker.init(frame, bbox)
                tracking = True
                bx, by, bw, bh = bbox
                smooth_cx = bx + bw / 2.0
                smooth_cy = by + bh / 2.0
                print(f"[TRACKER] Lock Target Initialized: {bbox}")
            except Exception as e:
                print(f"[ERROR] Gagal initialize tracker: {e}")
                tracking = False

            selected_bbox = None

        # ----------------------------------------------------
        # 4. MODUS TRACKING (LOCKED TARGET) vs DETEKSI MULTI-TARGET
        # ----------------------------------------------------
        if tracking and tracker is not None:
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                valid = (
                    w > 5
                    and h > 5
                    and x + w > 0
                    and y + h > 0
                    and x < frame_w
                    and y < frame_h
                )

                if valid:
                    target_cx = x + w / 2.0
                    target_cy = y + h / 2.0

                    # EMA Smoothing
                    if smooth_cx is None:
                        smooth_cx = target_cx
                    if smooth_cy is None:
                        smooth_cy = target_cy

                    smooth_cx = ALPHA * target_cx + (1.0 - ALPHA) * smooth_cx
                    smooth_cy = ALPHA * target_cy + (1.0 - ALPHA) * smooth_cy

                    # Perhitungan Error
                    error_x = smooth_cx - frame_cx
                    error_y = smooth_cy - frame_cy
                    error_x_control = apply_deadzone(error_x, DEADZONE)
                    error_y_control = apply_deadzone(error_y, DEADZONE)

                    # Perhitungan Jarak & Sudut (dari visionTarget.py)
                    distance = np.sqrt(
                        (smooth_cx - frame_cx) ** 2
                        + (smooth_cy - frame_cy) ** 2
                    )
                    cX_corr = smooth_cx - frame_cx
                    cY_corr = smooth_cy - frame_cy
                    angle = np.degrees(np.arctan2(cY_corr, cX_corr))
                    if angle < 0:
                        angle += 360

                    # Error Ternormalisasi (-1.0 s/d +1.0)
                    norm_x = np.clip(error_x / (frame_w / 2.0), -1.0, 1.0)
                    norm_y = np.clip(error_y / (frame_h / 2.0), -1.0, 1.0)

                    # Visualisasi Target Terkunci (GREEN BOX)
                    cv.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
                    target_center = (int(smooth_cx), int(smooth_cy))
                    cv.circle(frame, target_center, 6, (0, 0, 255), -1)
                    cv.line(
                        frame,
                        (frame_cx, frame_cy),
                        target_center,
                        (0, 255, 0),
                        2,
                    )

                    # Visualisasi Deadzone Box
                    dz_x, dz_y = int(DEADZONE), int(DEADZONE)
                    cv.rectangle(
                        frame,
                        (frame_cx - dz_x, frame_cy - dz_y),
                        (frame_cx + dz_x, frame_cy + dz_y),
                        (255, 255, 255),
                        1,
                    )

                    # Telemetri HUD
                    cv.putText(
                        frame,
                        "STATUS: LOCKED",
                        (20, 35),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"CX: {smooth_cx:.1f}  CY: {smooth_cy:.1f}",
                        (20, 65),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"ERR X: {error_x_control:.1f}  ERR Y: {error_y_control:.1f}",
                        (20, 90),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"NORM X: {norm_x:+.2f}  NORM Y: {norm_y:+.2f}",
                        (20, 115),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv.putText(
                        frame,
                        f"DIST: {distance:.1f} px  ANGLE: {angle:.1f} deg",
                        (20, 140),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    print(
                        f"\r[LOCK] Target=({smooth_cx:.1f}, {smooth_cy:.1f}) Dist={distance:.1f} Angle={angle:.1f} Error=({error_x_control:.1f}, {error_y_control:.1f})",
                        end="",
                    )
                else:
                    tracking = False
            else:
                # Target Hilang
                cv.putText(
                    frame,
                    "TARGET LOST (Tekan 'R' untuk re-select)",
                    (20, 35),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            # ------------------------------------------------
            # DETEKSI MULTI-TARGET UNLOCKED STATE
            # ------------------------------------------------
            # cv.putText(
            #     frame,
            #     f"MULTI-TARGET DETECTED ({len(detected_targets)}) - Click / Key (1-9) to Lock",
            #     (20, 35),
            #     cv.FONT_HERSHEY_SIMPLEX,
            #     0.65,
            #     (0, 255, 255),
            #     2,
            # )

            # Gambar semua target yang terdeteksi
            for t in detected_targets:
                tx, ty, tw, th = t["bbox"]
                tid = t["id"]
                tcx, tcy = t["center"]

                # Box target warna Kuning/Cyan
                cv.rectangle(
                    frame, (tx, ty), (tx + tw, ty + th), (255, 255, 0), 2
                )
                cv.circle(frame, (tcx, tcy), 4, (0, 255, 255), -1)

                # Label ID Target
                label = f"Target #{tid}"
                cv.putText(
                    frame,
                    label,
                    (tx, max(20, ty - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )

        # ----------------------------------------------------
        # VISUALISASI SELEKSI DRAG MOUSE
        # ----------------------------------------------------
        if selecting:
            x1, y1 = min(start_x, current_x), min(start_y, current_y)
            x2, y2 = max(start_x, current_x), max(start_y, current_y)
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # ----------------------------------------------------
        # HITUNG FPS
        # ----------------------------------------------------
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        cv.putText(
            frame,
            f"FPS: {fps:.1f}",
            (frame_w - 130, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Tampilkan Window Frame dan Mask
        cv.imshow(window_name, frame)
        # cv.imshow("HSV Mask", mask)

        # ----------------------------------------------------
        # INPUT KEYBOARD
        # ----------------------------------------------------
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("r") or key == ord("R"):
            print("\n[RESET] Unlock target, kembali ke mode deteksi multi-target.")
            tracker = None
            tracking = False
            smooth_cx, smooth_cy = None, None
        elif ord("1") <= key <= ord("9") and not tracking:
            target_idx = key - ord("1")
            if target_idx < len(detected_targets):
                selected_target = detected_targets[target_idx]
                selected_bbox = selected_target["bbox"]
                print(
                    f"\n[LOCK] Target #{selected_target['id']} dipilih via Tombol Keyboard '{chr(key)}'!"
                )

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()