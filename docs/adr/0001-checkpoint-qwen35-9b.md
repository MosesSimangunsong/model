# ADR-0001: Checkpoint `Qwen/Qwen3.5-9B`

- **Status:** Diterima
- **Tanggal:** 2026-11-02 (mencatat keputusan yang sudah dibuat sebelumnya)
- **Pemicu tinjau ulang:** Hanya bila pengguna/pembimbing memutuskan mengganti model.

## Konteks
Penelitian membutuhkan model yang membaca teks dan gambar lalu menghasilkan soal dan pembahasan terstruktur. Proposal V1 §2.4.3 telah memilih Qwen3.5-9B.

## Keputusan
Memakai checkpoint **`Qwen/Qwen3.5-9B`** (post-trained, bukan `Qwen3.5-9B-Base`) sebagai model dasar untuk semua kondisi eksperimen. Salinan lokal: `Model/Qwen3.5-9B`.

## Alasan
- Model multimodal berukuran ~9B: cukup besar untuk tugas penalaran, masih dapat di-fine-tune dengan adapter pada satu GPU.
- Lisensi Apache-2.0 menurut model card.

## Fakta terverifikasi (dari checkpoint, 2 November 2026)
- `Qwen3_5ForConditionalGeneration`, vision encoder 27 blok, bagian bahasa 32 lapisan **hibrida**: 24 linear attention + 8 full attention (lapisan 3, 7, …, 31).
- Konfigurasi LoRA teks di Colab sebelumnya hanya mengenai 8 lapisan full-attention (557.056 parameter = 8 × 69.632). Rincian di skill `qwen35-vlm-finetune` → `references/arsitektur-checkpoint.md`.

## Konsekuensi
- Agent tidak boleh mengusulkan model lain.
- Klaim arsitektur keluarga Qwen lain (Qwen2.5-VL, Qwen3-VL) tidak otomatis berlaku.
- Model ini tidak muat di GPU laptop 6 GB → lihat ADR-0012.

## Rujukan
- [Model card](https://huggingface.co/Qwen/Qwen3.5-9B); proposal V1 §2.4.3; `Model/Qwen3.5-9B/config.json`.
