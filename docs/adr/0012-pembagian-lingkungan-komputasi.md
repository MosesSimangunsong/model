# ADR-0012: Pembagian lingkungan komputasi

- **Status:** Sementara
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Jenis GPU Colab dikonfirmasi; akses GPU IT Del diberikan (lihat juga ADR-0003).

## Konteks
- Laptop: Windows 11, RTX 3060 Laptop **6 GB** — tidak cukup memuat Qwen3.5-9B 4-bit (~7,35 GiB di T4 hanya untuk memuat).
- Google Colab: dapat diakses; jenis GPU dan langganan belum dikonfirmasi.
- GPU IT Del (B200): belum dapat diakses; akan tersedia kemudian.

## Keputusan
| Lingkungan | Dipakai untuk |
|---|---|
| Laptop | Prapemrosesan (FFmpeg, ASR), anotasi, menulis/debug kode, pemeriksaan tokenisasi tanpa memuat bobot |
| Colab | Inferensi dan training pilot 9B |
| GPU IT Del | Eksperimen utama setelah akses tersedia |

Kode ditulis **portabel**: path dari satu konfigurasi (`pathlib`), deteksi Colab otomatis, deteksi dukungan bf16 (T4 tidak mendukung → fp16). Spesifikasi GPU aktual dicek dan dicatat di awal setiap sesi sebelum menentukan konfigurasi.

## Konsekuensi
- Pilot awal tidak bergantung pada GPU institusi.
- Hasil dari GPU berbeda dicatat dengan identitas perangkatnya; konfigurasi eksperimen utama dibekukan pada satu lingkungan.

## Rujukan
- `Pertanyaan_Skill_VLM.md` C2; skill `qwen35-vlm-finetune` → `references/lingkungan.md`.
