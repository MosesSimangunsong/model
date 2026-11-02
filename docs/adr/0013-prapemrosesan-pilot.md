# ADR-0013: Prapemrosesan pilot — FFmpeg dan faster-whisper

- **Status:** Sementara (untuk pilot; konfigurasi final ditentukan di skill prapemrosesan)
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Kualitas transkrip/frame pilot tidak memadai; skill `levelearn-preprocess` dibuat.

## Keputusan (yang sudah dijalankan untuk pilot)
- **Pemotongan:** FFmpeg (winget `Gyan.FFmpeg`), segmen 04:10–06:27 (250–387 dtk) di-encode ulang (libx264 CRF 18) agar titik potong akurat → `Data/Pilot/segmen_0410_0627.mp4`. Video sumber tidak diubah.
- **Frame:** FFmpeg `fps=1/5`, JPEG `-q:v 2` → 27 frame 1804×1080. Timestamp relatif = (N−1)×5; absolut = relatif + 250.
- **ASR:** faster-whisper (implementasi CTranslate2 dari model Whisper OpenAI), ukuran `small`, `int8`, **CPU** (cuBLAS CUDA tidak tersedia di laptop), `vad_filter=True` → `Data/Pilot/transkrip.{txt,json}`; bahasa terdeteksi `id`.

## Alasan
- FFmpeg adalah alat standar pengolahan media; faster-whisper lebih ringan untuk CPU/GPU kecil dengan bobot model Whisper yang sama.
- Ukuran `small` dan jarak 5 detik adalah titik awal, bukan konfigurasi optimal.

## Konsekuensi
- Transkrip **belum diperiksa manusia**; sudah tampak salah dengar istilah (mis. "Knya" kemungkinan "Q-nya"). Wajib dikoreksi sebelum dipakai sebagai data final.
- Untuk reproduksibilitas, catat versi FFmpeg dan faster-whisper.

## Rujukan
- `scripts/transkrip.py`; skill `whisper`; [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper).
