# Sumber

| Sumber | Dipakai untuk | Versi/diakses |
|---|---|---|
| Checkpoint lokal `Model/Qwen3.5-9B` (`config.json`, `preprocessor_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, header `.safetensors`) | Semua fakta di `arsitektur-checkpoint.md` | Dibaca 2 November 2026 |
| [Qwen/Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) | Identitas checkpoint, lisensi Apache-2.0 | Diperiksa 2 November 2026 (menurut dokumen konteks) |
| [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune): `src/dataset/sft_dataset.py`, `src/dataset/data_utils.py`, `src/train/train_sft.py`, `src/model/load_model.py`, `scripts/finetune_lora.sh`, README "Training Notes" | Pola masking prompt/jawaban, scaffold `<think>` untuk Qwen3.5, `sdpa`, pengecualian `visual` dari LoRA, `llm_int8_skip_modules`, batasan QLoRA + vision di framework itu | Commit `6707820` (10 September 2026); hanya dibaca, tidak dijalankan |
| Repo LeveLearn di `src/`: `Backend-.../prisma/schema.prisma`, `prisma/seed.js`, `src/services/AssessmentService.js`; `FrontendWeb-.../src/dto/AssessmentDto.ts`, `src/pages/Assessment.tsx`; `Mobile-.../lib/model/assessment.dart`, `lib/service/chapter_service.dart` | Format penyimpanan soal (format-data.md §0) | Dibaca 2 November 2026 |
| Notebook `Latihan_Qwen_LeveLearn.ipynb` (menurut dokumen konteks) | Konfigurasi dan hasil LoRA teks sebelumnya | Hasil tersimpan; bukan eksekusi ulang |
| `docs/Proposal/Proposal Draft (V1 28 Oktober).docx.md` (§2.4.3–2.4.4, Bab III) | Desain 2×2, split 70/15/15, pembekuan konfigurasi, inferensi deterministik, Tabel 3.6 spesifikasi keluaran, Tabel 3.7 entitas usulan | Versi 28 Oktober; lebih lama dari catatan 31 Oktober dan Bab III revisi 1 November |
| `Konteks_Fine_Tuning_Qwen35_9B_LeveLearn_untuk_Claude.md` dan `Pertanyaan_Skill_VLM.md` | Keputusan penelitian dan jawaban pengguna | 2 November 2026 |
