# VisionFPV Tracking & Evaluation Suite (`visionTarget.py` & Standalone Web Studio)

Sistem Visi Komputer berkinerja tinggi berbasis **Python**, **C++**, dan **HTML5/JS Standalone Studio** yang dirancang untuk deteksi, seleksi, simulasi fisika oklusi, dan pelacakan target visual (*multi-target locking*) secara real-time. Sistem ini sangat cocok digunakan untuk proyek robotik, FPV drone tracking, gimbal targeting, dan pemrosesan citra interaktif.

---

## Daftar Fitur Utama

### 1. VisionFPV Standalone Web Studio (`index.html`)
* **100% Serverless Client-Side App**: Berjalan langsung di browser tanpa perlu menyalakan server backend.
* **Desain UI Ringan & Bebas Emoticon**: Menggunakan palet warna kontras rendah yang nyaman dipandang (*Light Mode Minimalist UI*).
* **Terintegrasi Registry Algoritma C++ & Python**: Terhubung ke folder `vision_programs/` yang menampung 7 program pelacak visual (C++ SiamRPN, Kalman C++, vision_fpv3 C++, Python visionFPV3, visionFPV2, dll).

### 2. Fisika Oklusi Obstacle Solid & Ekstrapolasi Buta (Dead Reckoning)
* **Visual Block 100% Solid Opaque (`#334155`)**: Target yang berada di belakang obstacle tertutup rapat secara visual 100% dari kamera.
* **Unbiased Dead Reckoning**: Saat terhalang (`isOccluded = true`), algoritma tidak tahu jika target berbelok di dalam rintangan. Kotak pelacak mengekstrapolasi lurus menggunakan kecepatan terakhir sebelum terhalang $(lastVx, lastVy)$.
* **Penurunan IoU & Lost Lock Realistis**: Jika target berbelok di dalam rintangan solid, IoU turun secara alami. Jika $IoU < 0.3$, status otomatis berubah menjadi `[LOST LOCK]` (kotak merah putus-putus).
* **Uji Re-Acquisition**: Menguji kemampuan algoritma (seperti SiamRPN C++) dalam menangkap kembali target begitu keluar dari rintangan.

### 3. Editor Peta Interaktif & Custom Obstacle
* **Waypoint Flight Editor**: Menentukan titik jalur terbang target secara interaktif di canvas (Waypoint, Wild Evasive Random, Zigzag, Smooth, Circular).
* **Profil Kecepatan Penerbangan**: Mengatur profil kecepatan (Constant, Accelerating, Decelerating, Bell Curve).
* **Custom Obstacle Builder**: Membuat rintangan kustom berbentuk persegi yang bisa diputar (rotated rect) atau bentuk polygon 2D kustom (*dot-by-dot drawing*).
* **Import Gambar Kustom**: Pengunggah gambar latar belakang (*scenery background*) dan gambar objek target (*target sprite*).

### 4. Modular Drag & Drop Timeline Sequencer (0-300 Frames)
* Menggabungkan efek lingkungan per rentang frame:
  * **Camera Shake Jitter**: Simulasi getaran bodi drone.
  * **Motion Blur**: Simulasi efek gerakan cepat/buram.
  * **Speed Burst**: Simulasi percepatan tiba-tiba.
  * **Teleport / Lag Jump**: Simulasi *frame drop* akibat latensi sinyal transmisi FPV.

### 5. Tabel Evaluasi Ringkasan & Popup Modal Profile Radar 6-Aksis
* **Baris Tabel Interaktif (Clickable Rows)**: Mengklik baris algoritma pada tabel ringkasan langsung membuka **Popup Modal Profil Algoritma**.
* **Diagram Radar 6-Aksis (Game Character Stats)**: Diagram jaring laba-laba 6 parameter yang dinormalisasi (0-100):
  1. `Avg Latency` (Kecepatan respon latensi)
  2. `FPS Speed` (Kecepatan frame per detik)
  3. `RAM Efficiency` (Efisiensi penggunaan memori)
  4. `IoU Precision` (Presisi bentuk bounding box)
  5. `Lock Retention` (Tingkat ketahanan kuncian)
  6. `Overall Score` (Penilaian keseluruhan 0-100)
* **Badge Rekomendasi Teknis Presisi**:
  * `Edge MCU Real-Time` (C++, Latensi <= 2.0 ms, FPS >= 350)
  * `Companion Offboard Processing` (C++, Latensi <= 5.0 ms)
  * `Fault-Tolerant Lock Retention` (0 Lost frames, IoU >= 85%)
  * `High IoU Spatial Precision` (IoU >= 88%)
  * `High Latency Memory Bound` (Python/RAM tinggi)
  * `High Occlusion Sensitivity Risk` (Lost frames > 60)
  * `Balanced Real-Time Tracking` (Performa standar seimbang)

### 6. Hardware Execution Mode Toggle
* Beralih secara instan antara mode **Laptop Full Speed (x1.0)** dan **Raspberry Pi 4 Throttled (x3.8 Latency)** untuk menguji kinerja algoritma pada perangkat terbang terbatas.

---

## Persyaratan Sistem & Instalasi Koding Python/C++

### 1. Requirements Python
* Python 3.8+
* Paket Python:
  * `opencv-contrib-python`
  * `numpy`

### 2. Instalasi Dependency Python
```bash
pip install -r requirements.txt
```

### 3. Requirements C++
* G++ Compiler / GCC (Support C++17)
* OpenCV 4.x C++ Development Headers (`libopencv-dev`)

---

## Cara Penggunaan

### 1. Menjalankan Standalone Web Studio (Tanpa Server)
Buka file `index.html` atau `web/index.html` secara langsung di browser favorit Anda (Google Chrome / Mozilla Firefox / Edge):
```bash
# Buka via terminal Linux
xdg-open index.html
```

### 2. Menjalankan Server Web Python (Opsional)
Jika ingin menjalankan server HTTP ringan:
```bash
python3 web/app.py
# Atau jalankan script
bash web/run_server.sh
```

### 3. Menjalankan Skrip Pelacak Python (`visionTarget.py`)
```bash
python src/visionTarget.py
```

### 4. Menjalankan Executable C++ Tracker
```bash
cd cpp
make
./bin/siamrpn_tracker_cpp
```

### 5. Kontrol Keyboard & Mouse pada Program Python
| Aksi / Tombol | Fungsi |
| :--- | :--- |
| **Klik Kiri Mouse** | Mengunci target terdeteksi yang diklik |
| **Drag Kiri Mouse** | Membuat area pelacakan (ROI) manual kustom |
| **Tombol 1 - 9** | Mengunci target nomor ID #1 hingga #9 |
| **Tombol R** | Reset lock / melepas kuncian target |
| **Tombol ESC** | Keluar dari aplikasi |

---

## Struktur Direktori Proyek

```text
/home/fathir/kuyang/
├── README.md                             # Dokumentasi Utama Proyek
├── index.html                            # Standalone Web Benchmark Studio (Serverless)
├── requirements.txt                      # Dependency Python
├── vision_programs/                      # Registry Program Visi Komputer Centralized
│   ├── registry.json                     # Metadata Database Algoritma
│   ├── siamrpn_tracker_cpp.cpp
│   ├── kalman_tracker_cpp.cpp
│   ├── vision_fpv3_cpp.cpp
│   ├── visionFPV3_py.py
│   ├── visionFPV2_py.py
│   ├── visionFPV_py.py
│   └── visionTarget_py.py
├── src/                                  # Source Code Python & Drone API
├── cpp/                                  # Implementasi C++ & Makefile
├── web/                                  # Template & Web Server Application
├── benchmark/                            # Suite Benchmark & Pengujian Sintetik
├── data/                                 # Datasets & Hasil Benchmark
└── models/                               # Weights Model Deep Learning (ONNX)
```
