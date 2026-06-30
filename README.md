# Deteksi Emosi DeepFace + YOLO26 + Streamlit

Aplikasi Streamlit untuk mendeteksi emosi wajah dari:

- upload gambar,
- upload video lokal,
- link YouTube menggunakan `yt-dlp`,
- kamera realtime dari browser menggunakan `streamlit-webrtc`.

YOLO26 dipakai untuk menemukan target di frame, lalu DeepFace memprediksi emosi wajah. Secara default aplikasi memakai `yolo26n.pt` dari Ultralytics untuk mendeteksi `person`; DeepFace kemudian mencari wajah di dalam crop orang tersebut.

## Instalasi dengan Anaconda

```bash
conda env create -f environment.yml
conda activate deteksi-emosi
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

Saat pertama kali dijalankan, YOLO26 dan DeepFace dapat mengunduh model otomatis sehingga butuh koneksi internet.

## Cara pakai

1. Buka aplikasi Streamlit dari URL yang muncul di terminal.
2. Di sidebar, biarkan `YOLO weights / model` sebagai `yolo26n.pt` untuk mode default.
3. Pilih tab:
   - `Gambar` untuk upload foto,
   - `Video` untuk upload video lokal,
   - `YouTube` untuk memasukkan link YouTube,
   - `Realtime` untuk kamera browser.
4. Untuk video, atur `Analisis tiap N frame` agar proses tidak terlalu berat. Semakin besar nilainya, proses semakin cepat tetapi frame yang dianalisis lebih sedikit.
5. Di tab `Realtime`, pilih kamera `Depan` atau `Belakang`, klik `START`, lalu izinkan akses kamera dari browser.

## Menggunakan YOLO face model opsional

Default `yolo26n.pt` adalah model COCO yang mendeteksi `person`, bukan wajah langsung. Untuk hasil face bounding box yang lebih presisi, letakkan file model YOLO26 face kompatibel Ultralytics di folder `models/`, misalnya:

```text
models/yolo26n-face.pt
```

Lalu di sidebar:

- isi `YOLO weights / model` dengan `models/yolo26n-face.pt`,
- ubah `Tipe output YOLO` menjadi `Face model`.

## Catatan YouTube dan realtime

Fitur YouTube memakai `yt-dlp` dan membutuhkan koneksi internet. Gunakan hanya video yang memang Anda punya hak untuk unduh/proses. Aplikasi mengunduh single video file tanpa audio agar tidak perlu merge dengan `ffmpeg`.

Fitur realtime memakai kamera browser. Kamera biasanya hanya diizinkan browser pada `localhost` atau koneksi HTTPS. Jika membuka aplikasi dari HP lewat IP lokal seperti `http://192.168.x.x:8501`, browser akan menolak akses kamera karena halaman tidak secure.

Cara paling mudah untuk mencoba dari HP adalah memakai tunnel HTTPS, misalnya `ngrok`:

```bash
ngrok http 8501
```

Setelah itu buka URL `https://...ngrok-free.app` dari HP, lalu masuk ke tab `Realtime`, centang konfirmasi HTTPS, klik `START`, dan izinkan akses kamera.

## Troubleshooting

- Jika instalasi TensorFlow bermasalah, pastikan environment memakai Python 3.10 sesuai `environment.yml`.
- Jika proses video lambat, naikkan nilai `Analisis tiap N frame` atau turunkan `Maksimal frame dianalisis`.
- Jika DeepFace gagal mendeteksi wajah pada frame tertentu, aplikasi akan melewati frame/crop tersebut dan lanjut memproses frame berikutnya.
