# FPV Target Tracker v3.0 [C++ Edition]

Versi C++ berkinerja tinggi dari **Multi-Target & FPV Drone Vision Tracker** dengan dukungan optimasi memori, 2D Kalman Filter Motion Prediction, dan Telemetri Real-Time.

---

## 📋 Fitur Utama Versi C++

1. **Performa Tinggi & Zero-Copy Architecture**:
   - Menghilangkan overhead Python GIL dan alokasi memori dinamis NumPy.
   - Sangat dioptimasi untuk arsitektur CPU ARM NEON & TBB (Threading Building Blocks).
2. **2D Kalman Filter State Estimation**:
   - Prediksi lintasan target (`[x, y, vx, vy]^T`) saat objek mengalami oklusi atau buram akibat pergerakan cepat drone.
3. **Dual Tracker Engine**:
   - **CSRT Tracker** (`cv::TrackerCSRT`): Presisi tinggi untuk target dengan pergerakan kompleks.
   - **KCF Tracker** (`cv::TrackerKCF`): Kecepatan tinggi (100-200+ FPS di CPU).
4. **HUD & Control Signal**:
   - Vektor error terpolarisasi (`NORM X` & `NORM Y`: -1.0 s/d +1.0).
   - Pixel deadzone & perataan EMA.

---

## 🛠️ Persyaratan System & Dependency

- **C++14 Compiler** (`g++` >= 7.5 atau `clang`)
- **CMake** >= 3.10
- **OpenCV 4** (`libopencv-dev`)

### Instalasi Dependency (Linux / Raspberry Pi OS)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
```

---

## 🚀 Cara Kompilasi & Penggunaan

### 1. Kompilasi Proyek

Jalankan skrip build otomatis:

```bash
cd cpp
./build.sh
```

Biner terkompilasi akan berada di `cpp/build/vision_fpv3`.

### 2. Menjalankan Aplikasi

#### A. Menggunakan Webcam Kamera Lokal
```bash
./build/vision_fpv3 0
```

#### B. Menggunakan Stream Video Sintetik atau ESP32-CAM (MJPEG/RTSP)
```bash
./build/vision_fpv3 ../data/synthetic_fpv_test.mp4
```

---

## 🎮 Kontrol Keyboard & Mouse

| Aksikan / Tombol | Fungsi |
| :--- | :--- |
| **Klik Kiri Mouse** | Mengunci target titik yang diklik |
| **Drag Kiri Mouse** | Membuat ROI kotak pelacakan manual |
| **Tombol T** | Toggle Tracker Engine (CSRT <-> KCF) |
| **Tombol R** | Reset Tracker & menghapus kuncian |
| **Tombol ESC** | Keluar dari aplikasi |
