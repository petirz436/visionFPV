# FPV Target Tracker [C++ High-Performance Engine]

Versi C++ berkinerja tinggi dari **Multi-Target & FPV Drone Vision Tracker** dengan dukungan optimasi memori, 2D Kalman Filter Motion Prediction, SiamRPN Tracker C++, dan Telemetri Real-Time.

---

## Fitur Utama Versi C++

1. **Performa Ultra-Cepat & Zero-Overhead**:
   - Menghilangkan alokasi memori berlebih dan overhead Python GIL.
   - Eksekusi latensi ultra-rendah (0.8 - 1.8 ms per frame di PC / < 7 ms di Raspberry Pi 4).
2. **2D Kalman Filter State Estimation**:
   - Prediksi lintasan target (`[x, y, vx, vy]^T`) saat objek mengalami oklusi atau buram akibat pergerakan cepat drone.
3. **Multithreaded C++ Vision Suite (`vision_programs/`)**:
   - `siamrpn_tracker_cpp.cpp`: Tracker visual SiamRPN terkompilasi C++.
   - `kalman_tracker_cpp.cpp`: 2D Kalman Filter State Tracking.
   - `vision_fpv3_cpp.cpp`: Combined C++ Core Tracker.
4. **Terintegrasi Standalone Web Studio (`index.html`)**:
   - Hasil eksekusi C++ dapat diuji secara langsung di browser melalui Web Benchmark Studio.

---

## Persyaratan System & Dependency

- **C++17 Compiler** (`g++` >= 7.5 atau `clang`)
- **CMake** >= 3.10
- **OpenCV 4** (`libopencv-dev`)

### Instalasi Dependency (Linux / Raspberry Pi OS)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
```

---

## Cara Kompilasi & Penggunaan

### 1. Kompilasi Proyek

Jalankan kompilasi C++ dari direktori `cpp`:

```bash
cd cpp
make
```

### 2. Menjalankan Executable C++ Tracker

#### A. Menggunakan Webcam Kamera Lokal
```bash
./bin/siamrpn_tracker_cpp 0
```

#### B. Menggunakan Video Stream Sintetik FPV
```bash
./bin/siamrpn_tracker_cpp ../data/synthetic_fpv_test.mp4
```

---

## Kontrol Keyboard & Mouse

| Aksikan / Tombol | Fungsi |
| :--- | :--- |
| **Klik Kiri Mouse** | Mengunci target titik yang diklik |
| **Drag Kiri Mouse** | Membuat ROI kotak pelacakan manual |
| **Tombol T** | Toggle Tracker Engine |
| **Tombol R** | Reset Tracker & menghapus kuncian |
| **Tombol ESC** | Keluar dari aplikasi |
