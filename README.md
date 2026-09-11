# Multi-Target Lock & Vision Tracking System (`visionTarget.py`)

Sistem Visi Komputer berkinerja tinggi berbasis **Python** dan **OpenCV** yang dirancang untuk deteksi, seleksi, dan pelacakan target visual (*multi-target locking*) secara real-time. Sistem ini sangat cocok digunakan untuk proyek robotik, FPV drone tracking, gimbal targeting, dan pemrosesan citra interaktif.

---

## 📋 Daftar Fitur Utama

1. **Dual Video Input Source**:
   - **Webcam Lokal**: Mendukung pengambilan video langsung dari webcam internal/USB.
   - **ESP32-CAM Stream**: Mendukung penerimaan stream video MJPEG via HTTP/Wi-Fi dari modul ESP32-CAM dengan fitur auto-reconnect dan fallback handler.

2. **Multi-Target Detection (HSV Color Masking)**:
   - Otomatis memindai dan mendeteksi banyak objek sekaligus berdasarkan kontur warna HSV.
   - Dilengkapi penapis luas area minimal (`MIN_CONTOUR_AREA`) untuk memfilter *noise* latar belakang.

3. **CSRT Visual Object Tracking**:
   - Menggunakan algoritma **CSRT (Channel and Spatial Reliability Tracker)** yang sangat presisi untuk mengunci target meskipun objek bergerak, berubah bentuk, atau mengalami oklusi parsial.

4. **Interaksi Seleksi Target Fleksibel**:
   - **Klik Mouse**: Mengunci target yang terdeteksi di lokasi klik. Jika mengklik area kosong, otomatis membuat ROI (Region of Interest) di titik tersebut.
   - **Drag Mouse**: Membuat kotak ROI kustom secara manual pada area mana pun di layar.
   - **Shortcut Keyboard (1-9)**: Mengunci target terdeteksi secara instan berdasarkan nomor ID target.

5. **Filtering & Pengolahan Sinyal Kontrol**:
   - **EMA Smoothing (`ALPHA`)**: Perataan gerak target menggunakan *Exponential Moving Average* agar output koordinat tidak patah-patah.
   - **Pixel Deadzone Filter (`DEADZONE`)**: Toleransi piksel di sekitar titik pusat kamera untuk mencegah jitter sinyal kontrol.
   - **Perhitungan Telemetri Presisi**: Menghitung Jarak Euclidean, Sudut Polar (0°-360°), Error X/Y dalam piksel, serta **Normalized Error** (`-1.0` s/d `+1.0`) yang siap dihubungkan ke Flight Controller / PID Controller.

6. **Real-time HUD (Heads-Up Display)**:
   - Menampilkan crosshair pusat kamera, garis vektor ke target, ID target, kotak deadzone, indikator status lock/lost, serta counter FPS.

---

## 🛠️ Persyaratan Sistem & Instalasi

### 1. Requirements
* **Python 3.8+**
* Paket Python:
  * `opencv-contrib-python` (Diperlukan modul `legacy` / `contrib` untuk CSRT Tracker)
  * `numpy`

### 2. Instalasi Dependency
Jalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Penggunaan

### 1. Pengaturan Sumber Kamera
Buka file `src/visionTarget.py` dan sesuaikan parameter berikut pada bagian `# CONFIGURATION`:

#### A. Menggunakan Webcam Lokal
```python
USE_ESP_CAM = False
CAMERA_INDEX = 0    # Index webcam (0 untuk webcam bawaan, 1/2 untuk USB Cam)
```

#### B. Menggunakan ESP32-CAM Stream
```python
USE_ESP_CAM = True
ESP_CAM_URL = "http://192.168.1.100:81/stream"  # Sesuaikan dengan IP ESP32-CAM Anda
```

### 2. Menjalankan Program
Jalankan skrip utama dari direktori akar proyek:
```bash
python src/visionTarget.py
```

### 3. Menjalankan Evaluation Benchmark
Untuk melakukan benchmark perbandingan tracker (V1 vs V2 vs V3):
```bash
python benchmark/benchmark_suite.py --headless
python benchmark/benchmark_plotter.py
```

### 3. Kontrol Navigasi & Keyboard
| Aksikan / Tombol | Fungsi |
| :--- | :--- |
| **Klik Kiri Mouse** | Mengunci target terdeteksi yang diklik |
| **Drag Kiri Mouse** | Membuat area pelacakan (ROI) manual kustom |
| **Tombol 1 - 9** | Mengunci target nomor ID #1 hingga #9 |
| **Tombol R** | Reset lock / melepas kuncian target dan kembali ke mode deteksi multi-target |
| **Tombol ESC** | Keluar dari aplikasi |

---

## 🎛️ Panduan Tuning Parameter

Semua konfigurasi utama terletak di bagian atas file `visionTarget.py`:

```python
# ============================================================
# CONFIGURATION
# ============================================================
USE_ESP_CAM = False
ESP_CAM_URL = "http://192.168.1.100:81/stream"
CAMERA_INDEX = 0
DEFAULT_ROI_W = 100
DEFAULT_ROI_H = 100
ALPHA = 0.35            # Factor smoothing EMA
DEADZONE = 20           # Pixel deadzone error
MIN_CONTOUR_AREA = 500  # Filter kontur minimal untuk deteksi target

# HSV Color Mask (Contoh: Biru)
HSV_LOWER = np.array([90, 100, 100])
HSV_UPPER = np.array([130, 255, 255])
```

### 1. Tuning Warna Target (HSV Mask)
Ubah nilai `HSV_LOWER` dan `HSV_UPPER` sesuai warna objek yang ingin dideteksi:
* **Warna Biru (Default)**:
  ```python
  HSV_LOWER = np.array([90, 100, 100])
  HSV_UPPER = np.array([130, 255, 255])
  ```
* **Warna Hijau**:
  ```python
  HSV_LOWER = np.array([35, 100, 100])
  HSV_UPPER = np.array([85, 255, 255])
  ```
* **Warna Merah**:
  ```python
  HSV_LOWER = np.array([0, 100, 100])
  HSV_UPPER = np.array([25, 255, 255])
  ```

> 💡 **Tip Tuning HSV**:
> Nilai **Hue (H)** berkisar dari `0 - 179`, **Saturation (S)** `0 - 255`, dan **Value/Brightness (V)** `0 - 255`. Jika kondisi ruangan redup, turunkan batas bawah Value (misal `V_LOWER = 50`).

### 2. Tuning Sensitivitas Objek (`MIN_CONTOUR_AREA`)
* Jika bayangan atau titik kecil ikut terdeteksi sebagai target: **Naikkan** `MIN_CONTOUR_AREA` (misal ke `800` atau `1200`).
* Jika target berukuran kecil atau berada jauh dari kamera tidak terdeteksi: **Turunkan** `MIN_CONTOUR_AREA` (misal ke `200` atau `300`).

### 3. Tuning Perataan Gerak (`ALPHA` EMA)
* **`ALPHA` mendekati `1.0` (misal `0.8`)**: Respon pergerakan sangat cepat dan tanpa lag, namun sinyal koordinat lebih bergetar (*jittery*).
* **`ALPHA` mendekati `0.0` (misal `0.15`)**: Pergerakan sangat halus (*smooth*), tetapi memiliki sedikit keterlambatan (*latency/lag*) dalam mengikuti objek cepat.
* **Rekomendasi**: Nilai `0.30` - `0.40` memberikan keseimbangan terbaik untuk pelacakan drone/gimbal.

### 4. Tuning Deadzone (`DEADZONE`)
* `DEADZONE` menentukan batas area netral (dalam piksel) di sekitar pusat kamera.
* Jika error koordinat berada di dalam area `DEADZONE`, sinyal error kontrol akan bernilai `0.0`. Ini mencegah servo/motor bergetar terus-menerus saat target sudah berada di tengah.

---

## 🔧 Panduan Pemeliharaan & Troubleshooting (Maintenance)

### 1. Masalah Stream ESP32-CAM Terputus atau Lag
* **Gejala**: FPS rendah, frame patah-patah, atau error `Stream error / frame dropped`.
* **Solusi**:
  1. Pastikan ESP32-CAM terhubung ke jaringan Wi-Fi yang sama dengan komputer.
  2. Atur resolusi kamera pada firmware ESP32-CAM ke `FRAMESIZE_QVGA` (320x240) atau `FRAMESIZE_VGA` (640x480) untuk mendapatkan throughput 25-30 FPS.
  3. Periksa port HTTP stream ESP32-CAM (umumnya port `:81/stream` atau `:80/mjpeg`).

### 2. Error `CSRT tidak tersedia. Install opencv-contrib-python.`
* **Penyebab**: OpenCV standar (`opencv-python`) tidak menyertakan modul algoritma pelacak CSRT.
* **Solusi**: Uninstall versi standar dan install versi `contrib`:
  ```bash
  pip uninstall opencv-python opencv-python-headless
  pip install opencv-contrib-python
  ```

### 3. Target Mudah Lepas / Status `TARGET LOST`
* **Penyebab**: Perubahan intensitas cahaya ekstrim atau objek bergerak terlalu cepat keluar dari bingkai frame.
* **Solusi**:
  1. Tekan tombol **`R`** untuk melepaskan kuncian, lalu klik ulang target.
  2. Lakukan tuning ulang rentang warna `HSV_LOWER` & `HSV_UPPER` sesuai dengan pencahayaan lokasi saat ini.

---

## 📌 Struktur Proyek

```text
/home/fathir/kuyang/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                                  # Source Code Utama Pelacak & Drone API
│   ├── visionTarget.py                   # Multi-Target Vision Lock System (Main)
│   ├── visionFPV.py                      # Single-Target CSRT Tracker (v1)
│   ├── visionFPV2.py                     # CSRT Tracker + Kalman Filter (v2)
│   ├── visionFPV3.py                     # DaSiamRPN ONNX + Kalman Filter (v3)
│   ├── visionEksbot2.py                  # Auxiliary Vision Utility
│   └── droneAPI.py                       # MAVROS / ROS Drone Control Interface
├── benchmark/                            # Suite Benchmark & Pengujian Sintetik
│   ├── benchmark_suite.py                # Tracker Performance Evaluator
│   ├── benchmark_plotter.py              # Statistical Data Plotter
│   └── synthetic_generator.py            # Synthetic Test Video & GT Generator
├── data/                                 # Datasets & Output Files
│   ├── synthetic_gt.json
│   ├── synthetic_fpv_test.mp4
│   ├── benchmark_results.csv
│   ├── benchmark_summary.json
│   ├── benchmark_summary.png
│   └── benchmark_visualization.mp4
├── models/                               # Deep Learning / ONNX Model Weights
│   ├── dasiamrpn_model.onnx
│   ├── dasiamrpn_kernel_r1.onnx
│   └── dasiamrpn_kernel_cls1.onnx
└── docs/                                 # Dokumentasi & Media
    └── assets/
        └── Multi-Target Vision Lock System_screenshot_10.09.2026.png
```
