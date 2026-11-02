# Architecture Decision Records (ADR) — fine-tuning LeveLearn

ADR mencatat **keputusan** penelitian dan teknis beserta konteks, alasan, dan konsekuensinya, supaya alasan setiap pilihan dapat ditelusuri saat menulis dokumen TA.

## Status

| Status | Arti |
|---|---|
| **Diterima** | Keputusan berlaku. |
| **Sementara** | Berlaku untuk pilot; akan ditinjau ulang dengan pemicu yang disebutkan. |
| **Diusulkan** | Belum diputuskan; menunggu pilot atau persetujuan. |
| **Digantikan oleh ADR-xxxx** | Tidak berlaku lagi. |

Aturan:
- ADR tidak diedit untuk mengubah keputusan. Buat ADR baru yang menggantikannya, lalu ubah status ADR lama.
- Bedakan **keputusan** (ADR) dari **hasil eksperimen** (log/laporan). ADR boleh merujuk bukti, tetapi tidak mengklaim hasil yang belum ada.
- Bila dokumen proposal berbeda dari ADR yang lebih baru, ADR menjadi acuan sampai proposal diperbarui (lihat kolom "Sinkron dokumen").

## Daftar

| No | Judul | Status | Sinkron dokumen |
|---|---|---|---|
| [0001](0001-checkpoint-qwen35-9b.md) | Checkpoint `Qwen/Qwen3.5-9B` | Diterima | Sesuai proposal V1 |
| [0002](0002-keluaran-qg-dan-ae.md) | Keluaran hanya QG dan AE, dua tahap | Diterima | **Proposal V1 perlu diperbarui** (masih menyebut "materi penjelasan") |
| [0003](0003-qlora-untuk-pilot.md) | QLoRA 4-bit untuk pilot | Sementara | **Proposal V1 perlu diselaraskan** |
| [0004](0004-vision-encoder-dibekukan.md) | Vision encoder dibekukan, LoRA di bagian bahasa | Sementara | Belum dibahas eksplisit di proposal |
| [0005](0005-target-modul-lora.md) | Target modul LoRA | Diusulkan | Proposal: ditentukan lewat validasi |
| [0006](0006-masukan-frame-transkrip-instruksi.md) | Masukan: frame terurut + transkrip ASR + instruksi | Diterima | Sesuai proposal V1 |
| [0007](0007-implementasi-transformers-peft.md) | Implementasi Transformers + PEFT (script sendiri) | Diterima | — |
| [0008](0008-thinking-mode-mati.md) | `enable_thinking=False` | Diterima | — |
| [0009](0009-format-keluaran-json.md) | Keluaran JSON, kunci berupa huruf, metadata dari kode | Diusulkan | Sejalan dengan Tabel 3.6 proposal V1 |
| [0010](0010-bahasa-keluaran-indonesia.md) | Soal dan pembahasan berbahasa Indonesia | Diterima | — |
| [0011](0011-split-per-kelompok-sumber.md) | Split per kelompok sumber | Diterima | Sesuai proposal V1 §3.4.2 |
| [0012](0012-pembagian-lingkungan-komputasi.md) | Pembagian lingkungan: laptop, Colab, GPU IT Del | Sementara | — |
| [0013](0013-prapemrosesan-pilot.md) | Prapemrosesan pilot: FFmpeg dan faster-whisper | Sementara | — |
| [0014](0014-aturan-kerja-agent.md) | Aturan kerja agent AI di repo | Diterima | — |
| [0015](0015-kontrak-prompt-json-qg-ae-v01.md) | Kontrak prompt dan JSON QG/AE v0.1 (melengkapi nama field ADR-0009) | Sementara | **Proposal V1 Tabel 3.6 perlu diselaraskan** (stimulus, status, evidence_ids, struktur AE) |

## Templat

```markdown
# ADR-XXXX: <judul>

- **Status:** Diterima | Sementara | Diusulkan | Digantikan oleh ADR-YYYY
- **Tanggal:** YYYY-MM-DD
- **Pemicu tinjau ulang:** <kondisi yang membuat keputusan ini perlu dibuka lagi>

## Konteks
## Keputusan
## Alasan
## Alternatif yang dipertimbangkan
## Konsekuensi
## Rujukan
```
