# ADR-0004: Vision encoder dibekukan, LoRA pada bagian bahasa

- **Status:** Sementara
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Hasil pilot (mis. formula pada frame tidak terbaca dengan baik) dan ketersediaan GPU IT Del.

## Konteks
Qwen3.5-9B memiliki vision encoder (`model.visual`, 27 blok). Melatihnya menambah kebutuhan memori, dan menurut README 2U1 tidak dapat digabung dengan QLoRA di framework tersebut.

## Keputusan
Untuk pilot: **bobot vision encoder dan merger dibekukan**. Adapter LoRA hanya dipasang pada bagian bahasa. Frame **tetap diproses** oleh vision encoder dan memengaruhi prediksi — pelatihan tetap multimodal.

## Alasan
- Hemat memori dan sejalan dengan ADR-0003.
- Batasan dari README 2U1 diperlakukan sebagai batas implementasi framework itu, bukan larangan universal.

## Konsekuensi
- Pemeriksaan wajib: tidak ada parameter `visual` dengan `requires_grad=True`; batch berisi `pixel_values` yang benar.
- Strategi pembekuan dilaporkan eksplisit di TA.
- Eksperimen utama dapat meninjau ulang keputusan ini; perubahan dicatat sebagai ADR baru.

## Rujukan
- `Pertanyaan_Skill_VLM.md` C3; README 2U1 "Training Notes".
