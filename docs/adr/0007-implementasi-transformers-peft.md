# ADR-0007: Implementasi training dengan Transformers + PEFT (script sendiri)

- **Status:** Diterima
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Hambatan teknis yang tidak dapat diselesaikan dengan pendekatan ini.

## Keputusan
Menulis script training sendiri dengan **Transformers + PEFT**. [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) (commit `6707820`) dipakai sebagai **acuan bacaan**, tidak diinstal/dijalankan. TRL `SFTTrainer` hanya dipakai setelah penanganan data multimodalnya terverifikasi pada versi terpasang.

## Alasan
- Setiap bagian kode harus dapat dijelaskan di dokumen TA.
- Mengurangi ketergantungan pada kode pihak ketiga yang berisi *monkey patch*.

## Alternatif
- Framework 2U1 (mendukung Qwen3.5, tetapi kode pihak ketiga); Unsloth dan LLaMA-Factory (dukungan Qwen3.5 belum diverifikasi).

## Konsekuensi
- Collator, masking label, dan pemeriksaan gambar ditulis dan diuji sendiri (uji 1 batch wajib).
- Pola masking mengikuti 2U1 untuk Qwen3.5: scaffold `<think>\n\n</think>\n\n` termasuk prompt (label `-100`).

## Rujukan
- `Pertanyaan_Skill_VLM.md` C1; skill `qwen35-vlm-finetune` → `references/pipeline-training.md`.
