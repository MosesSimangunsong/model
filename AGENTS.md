# AGENTS.md — LeveLearn: fine-tuning Qwen3.5-9B (QG + AE)

Instruksi untuk agent AI (Claude Code, Codex, dll.) di repo ini. **Di awal sesi, baca [docs/STATUS.md](docs/STATUS.md)** (progres & langkah berikutnya); perbarui di akhir sesi kerja. Konteks lengkap: [Konteks_Fine_Tuning_Qwen35_9B_LeveLearn_untuk_Claude.md](Konteks_Fine_Tuning_Qwen35_9B_LeveLearn_untuk_Claude.md) — baca bagian yang relevan sebelum mengubah alur penelitian.

## Proyek

Tugas Akhir: membangkitkan soal pilihan ganda (QG) dan pembahasan jawaban (AE) dari segmen video Logika Proposisi, dengan masukan **frame terurut + transkrip ASR + instruksi**, lewat QLoRA pada `Qwen/Qwen3.5-9B` (post-trained, bukan Base). Model sudah ditetapkan; jangan usulkan ganti model.

- QG: 1 panggilan/segmen → 3 soal, masing-masing 4 opsi, tepat 1 kunci.
- AE: 1 panggilan/soal → pembahasan (alasan kunci, alasan tiap pengecoh, ID segmen + waktu dari metadata).
- Pengguna sedang belajar alur fine-tuning untuk menuliskannya di dokumen TA: jelaskan dalam bahasa Indonesia sederhana, ringkas, dengan istilah teknis dijelaskan saat pertama muncul.

## Lingkungan (sudah diverifikasi)

- Windows 11, PowerShell. Python venv di `.venv` (`.\.venv\Scripts\Activate.ps1`).
- GPU lokal: RTX 3060 Laptop **6 GB VRAM** — tidak cukup untuk memuat/melatih 9B (4-bit saja ~7,35 GiB di T4). Laptop dipakai untuk prapemrosesan, anotasi, dan menulis/debug kode; inferensi & training 9B di Colab atau GPU IT Del (akses B200 belum terverifikasi).
- Bobot model lokal: `Model/Qwen3.5-9B` (4 shard `.safetensors`).
- FFmpeg terpasang (winget `Gyan.FFmpeg`). faster-whisper terpasang; CUDA cuBLAS **tidak** ada → jalankan Whisper dengan `--device cpu`.

## Dokumen penelitian

- **[docs/adr/](docs/adr/README.md): keputusan proyek (ADR).** Ini acuan utama bila bertentangan dengan proposal. Keputusan baru dicatat sebagai ADR baru, ADR lama tidak diedit untuk mengubah keputusan.
- `docs/Proposal/Proposal Draft (V1 28 Oktober).docx.md` adalah proposal V1, dan sebagian sudah tertinggal dari ADR (lihat kolom "Sinkron dokumen" di daftar ADR). Bab III revisi 1 November belum ada di repo dan akan dibuat pengguna.
- Bila sumber-sumber bertentangan dan belum ada ADR yang memutuskan, jangan menyelesaikan konflik sendiri; tanyakan ke pengguna.
- Keluaran model **hanya QG dan AE** (ADR-0002). QLoRA hanya untuk pilot dan dapat berubah saat GPU IT Del tersedia (ADR-0003).

## Kode LeveLearn

`src/` berisi repo aplikasi LeveLearn (Backend Express/Prisma, FrontendWeb React, Mobile Flutter, `ml_service`, `essay_grader`). Soal disimpan di `Assessment.questions` (lihat skill `qwen35-vlm-finetune`, format-data.md §0). Jangan mengubah `src/` kecuali diminta.

## Data

- `Data/Video/` — video sumber. Jangan ditimpa atau diubah.
- `Data/Pilot/segmen_0410_0627.mp4` — pilot BDD, 250–387 dtk absolut (137 dtk).
- `Data/Pilot/frames/frame_NNN.jpg` — 27 frame, `fps=1/5`; timestamp relatif = (N−1)×5, absolut = relatif + 250.
- `scripts/transkrip.py` — ASR → `Data/Pilot/transkrip.{txt,json}` (sudah dibuat: Whisper `small`, bahasa terdeteksi `id`, 15 potongan; **belum diperiksa manusia**).

## Aturan penelitian (wajib)

- **Bedakan hasil nyata vs rencana.** Jangan mengarang hasil eksperimen, angka, spesifikasi, atau isi video. Klaim hanya yang sudah dijalankan dan dilihat keluarannya.
- Verifikasi terhadap checkpoint ini, jangan menebak: nama modul target LoRA, token visual, chat template, kelas model/processor. Arsitektur Qwen3.5 tidak boleh diasumsikan sama dengan Qwen/Llama lain.
- Training multimodal harus benar-benar membawa piksel gambar di batch; loss hanya pada token assistant (label prompt = `-100`); uji 1 batch forward/backward sebelum training panjang.
- Split train/val/test **per sumber/video**, bukan per contoh.
- Keluaran AI yang belum diperiksa manusia bukan ground truth. Simpan keluaran mentah terpisah dari hasil parsing.
- Penurunan loss bukan bukti peningkatan kualitas; bandingkan model dasar vs adapter pada data uji terpisah dengan prompt & konfigurasi generasi yang sama.
- Catat untuk reproduksibilitas: versi paket, seed, ID/revisi model, parameter prapemrosesan, peak VRAM.

## Cara kerja

1. **Pikir dulu.** Nyatakan asumsi; jika ambigu, tanya. Sebut pendekatan yang lebih sederhana bila ada.
2. **Sederhana.** Kode minimum yang menyelesaikan tugas; tanpa fitur/abstraksi spekulatif.
3. **Perubahan bedah.** Sentuh hanya yang diminta; ikuti gaya kode yang ada.
4. **Berbasis verifikasi.** Untuk tugas bertahap, tulis rencana singkat `langkah → cara cek`, lalu cek hasilnya sebelum lanjut.
5. Jalankan langkah bertahap; jangan menjalankan pipeline penuh saat path/sumber belum ada. Jangan unduh/instal hal besar (model, CUDA) tanpa persetujuan.
6. **Jangan pernah `git commit` atau `git push`.** Pengguna melakukannya sendiri; laporkan file yang diubah dan hasil pemeriksaan.
7. Jika butuh banyak informasi dari pengguna sebelum membangun sesuatu, kumpulkan pertanyaan dalam satu file `.md` untuk dijawab.
8. **Efektif dan efisien, tanpa mengurangi ketelitian.** Cari dulu (Grep/Glob), baca hanya bagian yang relevan, dan jangan membaca ulang hal yang sudah diketahui. Tetapi verifikasi tetap dilakukan sampai tuntas: periksa sumber asli, jalankan pemeriksaan, dan uji klaim penting. Hemat token tidak boleh menjadi alasan untuk menebak.

## Skill proyek (`.claude/skills/`)

- `qwen35-vlm-finetune` — **skill utama** untuk fine-tuning multimodal proyek ini (fakta checkpoint terverifikasi, kontrak data, pipeline, catatan TA).
- `peft`, `whisper`, `trl-training` — referensi umum, sumber di [.claude/skills/SOURCES.md](.claude/skills/SOURCES.md). Contoh kodenya generik (Llama/teks): daftar `target_modules` dan collator di skill `peft` tidak memasking prompt dan tidak multimodal — ikuti `qwen35-vlm-finetune` bila bertentangan.
