# ADR-0006: Masukan model — frame terurut + transkrip ASR + instruksi

- **Status:** Diterima (jumlah frame, resolusi, dan sampling masih ditentukan lewat pilot)
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Pilot menunjukkan konteks terlalu panjang atau notasi tidak terbaca.

## Konteks
Video pembelajaran menyampaikan informasi lewat ucapan dan visual (formula, tabel). Qwen tidak menerima audio; ucapan harus diubah menjadi teks oleh ASR. Model mendukung input video native, tetapi jalurnya belum diverifikasi.

## Keputusan
Masukan per pemanggilan: **beberapa frame terurut (gambar) + transkrip ucapan segmen + instruksi tugas** (QG atau AE; AE ditambah soal, opsi, kunci). Video native dicatat sebagai eksplorasi lanjutan.

Untuk desain 2 × 2 (proposal V1 §3.3.1) dan kondisi transkrip + OCR, penyusun prompt mendukung saklar **dengan/tanpa frame** dan **dengan/tanpa OCR**; bagian lain identik.

## Konsekuensi
- Satu frame 1804×1080 diperkirakan ~1.900 token visual tanpa pengecilan (27 frame ≈ 51.000 token) — **perkiraan**, diukur ulang dengan `image_grid_thw`. Jumlah frame/resolusi harus dibatasi dan keterbacaan formula diperiksa.
- Timestamp frame dibedakan antara relatif segmen dan absolut video.
- OCR hanya kondisi pembanding, bukan pengganti transkrip.

## Rujukan
- `Pertanyaan_Skill_VLM.md` C5; proposal V1 §3.3.1, §3.5.1; skill `qwen35-vlm-finetune`.
