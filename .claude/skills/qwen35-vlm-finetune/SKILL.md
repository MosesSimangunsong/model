---
name: qwen35-vlm-finetune
description: Fine-tuning multimodal (frame video + transkrip + instruksi) Qwen/Qwen3.5-9B dengan QLoRA untuk tugas QG dan AE LeveLearn. Gunakan saat menyiapkan format data latih, collator, masking label, pemilihan modul target LoRA, uji 1 batch, training, simpan/muat ulang adapter, atau saat menjelaskan alur fine-tuning untuk dokumen TA.
---

# Fine-tuning multimodal Qwen3.5-9B untuk QG/AE LeveLearn

Skill ini memandu implementasi **Transformers + PEFT** (script sendiri). Repo [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) hanya acuan bacaan, tidak dipasang atau dijalankan. TRL `SFTTrainer` boleh dipakai **hanya setelah** penanganan data multimodalnya diverifikasi pada versi yang terpasang.

Prapemrosesan (potong video, frame, ASR) dan evaluasi berada di luar skill ini dan akan menjadi skill terpisah. Kontrak data yang harus mereka hasilkan ada di [references/format-data.md](references/format-data.md).

## Status saat ini (perbarui bila berubah)

| Hal | Status |
|---|---|
| LoRA **teks** 12 langkah di Colab T4 | **Sudah dijalankan.** Adapter berubah; loss latih 2,218 → 2,140; generasi tidak berubah. Bukan bukti kualitas. |
| LoRA teks itu hanya mengenai 8 lapisan full-attention dari 32 lapisan bahasa | **Terverifikasi** dari bentuk tensor (lihat referensi arsitektur) |
| Segmen pilot, 27 frame, transkrip Whisper `small` (bahasa `id`) | **Sudah ada**; transkrip **belum diperiksa manusia** |
| Inferensi multimodal model dasar | Belum |
| Dataset QG/AE teranotasi dan diperiksa | Belum |
| Fine-tuning **multimodal** | Belum |
| Format soal backend LeveLearn (`src/`) | **Terverifikasi**: `Assessment.questions` = string JSON `[{question, options[], answer: teks opsi, type: "MC"\|"EY"}]`. **Tidak ada field pembahasan/segmen/waktu.** Lihat format-data.md §0 |
| Proposal V1 28 Oktober (`docs/Proposal/Proposal Draft (V1 28 Oktober).docx.md`) | **Dibaca** (2.4.3–2.4.4, Bab III). Rancangan yang relevan ada di bawah. |
| Bab III revisi 1 November (`Bab_III_Metodologi_LeveLearn_Lengkap_Revisi.docx`) | **Belum ditemukan/diperiksa**. Bila bertentangan dengan proposal V1, keputusan yang lebih baru menang. |

## Keputusan yang sudah ditetapkan pengguna

- Checkpoint `Qwen/Qwen3.5-9B` (lokal: `Model/Qwen3.5-9B`). Jangan usulkan model lain.
- QG dan AE adalah dua tahap pemanggilan; boleh satu adapter dengan instruksi berbeda.
- Masukan: **beberapa frame terurut + transkrip + instruksi**. Video native hanya eksplorasi lanjutan.
- QLoRA 4-bit untuk pilot; dapat berubah ke LoRA 16-bit saat GPU IT Del tersedia (ADR-0003).
- Vision encoder **dibekukan** untuk pilot; LoRA pada bagian bahasa. Gambar tetap wajib diproses. Batasan "QLoRA tidak bisa + vision training" dari README 2U1 adalah batas framework itu, bukan larangan universal.
- `enable_thinking=False`. AE tetap memuat pembahasan lengkap, termasuk alasan kunci dan alasan tiap pengecoh.
- Keluaran berupa JSON yang dapat dikonversi ke format soal LeveLearn. Soal dan pembahasan berbahasa Indonesia meski videonya berbahasa Inggris; notasi logika tidak diubah.
- Lingkungan: laptop (RTX 3060 6 GB, tidak cukup untuk 9B) dan Colab (jenis GPU belum dikonfirmasi). GPU IT Del menyusul. Kode harus portabel.
- **Jangan pernah commit atau push.** Laporkan file yang diubah dan hasil pemeriksaannya.

## Rancangan dari proposal V1 (status: Rencana)

- **Desain 2 × 2**: model dasar (M0) atau hasil fine-tuning (M1), masing-masing dengan frame (V) atau tanpa frame (T). Kodenya M0-V, M1-V, M0-T, M1-T; perbandingan utama M1-V vs M0-V. Rumusan masalah juga menyebut kondisi transkrip + OCR. Jadi fungsi penyusun prompt harus mendukung **dengan/tanpa frame dan dengan/tanpa OCR**, sementara bagian lainnya identik.
- Data latih memakai frame (multimodal). M1-T tetap memakai adapter yang sama.
- Split sekitar **70/15/15 per kelompok sumber**. Data pilot dari bagian pengembangan, tidak masuk uji akhir. Daftar split dan seed tidak diubah karena hasil uji.
- Hyperparameter (rank, learning rate, batch efektif, epoch, target modul, kebijakan berhenti) dipilih dengan data latih/validasi lalu **dibekukan sebelum uji**. Checkpoint dipilih berdasarkan kualitas validasi pada tugas-tugasnya, bukan loss saja.
- QLoRA dipakai untuk **pilot** karena keterbatasan GPU. **Keputusan ini dapat berubah** ke LoRA 16-bit setelah GPU IT Del tersedia ([ADR-0003](../../../docs/adr/0003-qlora-untuk-pilot.md)). Bila beralih, eksperimen utama dijalankan ulang dengan konfigurasi yang sama di semua kondisi. Presisi bobot dasar selalu dicatat di log.
- Inferensi evaluasi: **deterministik** (sampling mati), **respons pertama** dipakai, retry operasional dibatasi dan dicatat terpisah.
- Target segmen sekitar 3 menit (pilot membandingkan 2/3/4 menit). Segmen pilot BDD berdurasi 137 detik.
- Keluaran **hanya QG dan AE** (dikonfirmasi pengguna 2 November 2026, [ADR-0002](../../../docs/adr/0002-keluaran-qg-dan-ae.md)). "Materi penjelasan" di proposal V1 belum diperbarui dan **bukan** tugas model.

Semua keputusan beserta alasannya tercatat di [docs/adr/](../../../docs/adr/README.md). Bila ada keputusan baru, buat ADR baru (jangan mengedit keputusan lama).

## Alur kerja (setiap langkah: kerjakan → cek → baru lanjut)

1. **Cek lingkungan** → GPU, VRAM, dukungan bf16, versi paket. Lihat [references/lingkungan.md](references/lingkungan.md). Jangan tentukan konfigurasi sebelum ini.
2. **Verifikasi struktur model** di implementasi yang terpasang → `named_modules()` cocok dengan [references/arsitektur-checkpoint.md](references/arsitektur-checkpoint.md).
3. **Inferensi dasar multimodal** pada segmen pilot → cek `pixel_values` ada, jumlah token gambar sesuai `image_grid_thw`, keluaran tidak terpotong.
4. **Siapkan data** sesuai [references/format-data.md](references/format-data.md) → hanya contoh berstatus `diperiksa` yang masuk train; split per `source_group_id`.
5. **Bangun dataset + collator + masking** sesuai [references/pipeline-training.md](references/pipeline-training.md) → jalankan semua pemeriksaan sampel.
6. **Pasang LoRA** → cetak modul yang terkena dan jumlah parameter yang dapat dilatih; pastikan tidak ada parameter `visual` yang dilatih.
7. **Uji 1 batch forward/backward** → loss finite, gradien hanya pada LoRA, peak VRAM dicatat.
8. **Training pilot** → log loss latih/validasi, waktu, VRAM; simpan checkpoint.
9. **Simpan lalu muat ulang adapter dari sesi baru** → keluaran sama untuk masukan dan konfigurasi generasi yang sama.
10. **Serahkan ke evaluasi**: model dasar vs adapter pada data uji terpisah. Penurunan loss saja tidak cukup.

## Mode penjelasan untuk dokumen TA

Setelah setiap langkah, sertakan ringkasan **Apa** (yang dilakukan), **Mengapa** (alasannya), dan **Status** (salah satu label di bawah). Panduan dan contohnya ada di [references/catatan-ta.md](references/catatan-ta.md).

Label status yang wajib dibedakan:
- **Rencana** — konfigurasi yang dimaksudkan, belum dijalankan.
- **Diimplementasikan** — kode ada dan sudah dijalankan; sebutkan apa yang diamati.
- **Hasil** — angka/keluaran nyata dari eksekusi, dengan lokasi log.
- **Belum diketahui** — belum diverifikasi.

Jangan pernah menulis rencana seolah-olah sudah menjadi hasil.

## Larangan

- Mengarang angka, hasil, spesifikasi perangkat, nama modul, atau isi video.
- Menebak ID token visual atau nama modul; ambil dari config/processor/model.
- Menimpa video, model dasar, atau data asli. Tulis keluaran ke folder baru.
- Memakai keluaran AI yang belum diperiksa sebagai target latih.
- Menginstal paket besar atau CUDA tanpa persetujuan pengguna.

Sumber dan commit acuan ada di [references/sumber.md](references/sumber.md).
