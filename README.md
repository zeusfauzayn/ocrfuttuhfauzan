# NeoOCR - Sistem Pengenalan Tulisan Tangan Berbasis CNN

**NeoOCR** adalah sebuah aplikasi web yang dirancang untuk melakukan *Optical Character Recognition* (OCR) pada tulisan tangan, mencakup pengenalan **Angka (0-9)** dan **Huruf (A-Z)**. Sistem ini dibangun menggunakan metode *Deep Learning* dengan arsitektur **Convolutional Neural Network (CNN)** untuk mencapai akurasi pengenalan yang tinggi.

Proyek ini dikembangkan sebagai bagian dari tugas mata kuliah Pengolahan Citra Digital, mengimplementasikan konsep *Computer Vision* modern dalam antarmuka web yang interaktif.

---

## � Fitur Utama

Aplikasi ini dilengkapi dengan berbagai fitur teknis berikut:

*   **Pengenalan Karakter Cerdas:** Mampu mengidentifikasi karakter tulisan tangan (alfanumerik) dari citra digital.
*   **Preprocessing Citra Otomatis:** Mengimplementasikan teknik pengolahan citra digital meliputi:
    *   *Adaptive Thresholding:* Menangani variasi pencahayaan pada citra input.
    *   *Noise Reduction:* Menghilangkan derau (noise) untuk hasil segmentasi yang lebih bersih.
    *   *Character Segmentation:* Algoritma pemisahan karakter dan pengurutan baris secara otomatis.
*   **Mode Input Ganda:** Mendukung unggah (upload) file citra dan penulisan langsung pada kanvas digital.
*   **Antarmuka Modern:** Desain antarmuka pengguna (UI) berbasis *Glassmorphism* yang responsif.

---

## 💻 Spesifikasi Sistem

Untuk menjalankan aplikasi ini di lingkungan lokal (*Localhost*), dibutuhkan spesifikasi perangkat lunak sebagai berikut:

*   **Bahasa Pemrograman:** Python 3.8 - 3.11
*   **Framework Web:** Flask 3.0
*   **Library Utama:** TensorFlow, OpenCV, NumPy, Pillow

---

## � Panduan Instalasi dan Penggunaan

Berikut adalah langkah-langkah untuk menjalankan aplikasi NeoOCR:

### 1. Persiapan Lingkungan
Pastikan Python telah terinstal. Disarankan untuk menggunakan *Virtual Environment* agar dependensi proyek terisolasi dengan baik.

Buka terminal atau Command Prompt pada direktori proyek, lalu jalankan perintah:

**Windows:**
```powershell
.\venv\Scripts\activate
```

### 2. Instalasi Dependensi
Unduh dan pasang semua *library* yang dibutuhkan dengan perintah:

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Mulai server aplikasi dengan menjalankan perintah:

```bash
python app.py
```
Tunggu hingga muncul pesan *server running* pada terminal.

### 4. Akses Aplikasi
Buka peramban web (browser) dan kunjungi alamat berikut:
**http://127.0.0.1:5000**

---

## ☁️ Panduan Deployment (Opsional)

Aplikasi ini dirancang dengan arsitektur modular yang mendukung *Hybrid Hosting* untuk dapat diakses secara publik:

1.  **Backend (Model AI):** Dijalankan pada platform **Hugging Face Spaces** menggunakan Docker container untuk menangani komputasi neural network.
2.  **Frontend (Antarmuka):** Dapat diintegrasikan pada layanan hosting web standar (seperti InfinityFree atau RumahWeb) menggunakan file `index_hosting.html` yang telah disediakan.

---

## 🧠 Pelatihan Model

Sistem ini menggunakan model yang telah dilatih (`model/emnist_cnn.h5`). Apabila diperlukan pelatihan ulang model menggunakan dataset EMNIST, dapat menjalankan skrip berikut:

```bash
python train_emnist_manual.py
```

---

## 👤 Pengembang

**Nama:** Futtuh Fauzan  
**Proyek:** Sistem OCR Tulisan Tangan (Pengolahan Citra Digital)
**Link hasil hosting :** https://ocrfauzan.kesug.com/

---
*Dokumen ini disusun untuk melengkapi submisi tugas/proyek akademik.*
