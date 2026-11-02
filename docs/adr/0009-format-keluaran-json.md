# ADR-0009: Keluaran JSON; kunci berupa huruf; metadata disisipkan kode

- **Status:** Diusulkan (JSON terstruktur: Diterima; nama field dan pemetaan: Diusulkan)
- **Tanggal:** 2026-11-02
- **Pemicu keputusan:** Pilot parsing dan keputusan perubahan skema LeveLearn.

## Konteks
Format soal LeveLearn saat ini (diperiksa di `src/`): `Assessment.questions` = string JSON `[{question, options[], answer, type}]`, dengan `answer` berupa **teks opsi** yang dicocokkan secara teks di aplikasi mobile. **Tidak ada** field pembahasan, ID segmen, atau waktu. Proposal V1 Tabel 3.6 meminta soal dengan opsi A–D, kunci, konsep, dan ID segmen; Tabel 3.7 mengusulkan entitas baru (`VideoSegment`, `GeneratedQuestion`, `GeneratedExplanation`, …).

## Keputusan
- Keluaran QG dan AE berupa **JSON** yang divalidasi kode. (Diterima)
- Usulan: model menulis `answer_key` sebagai **huruf A–D**; kode mengonversi ke `answer` berupa teks opsi untuk LeveLearn dan memastikan opsi unik. (Diusulkan)
- `segment_id` dan waktu awal–akhir **disisipkan kode** dari metadata, tidak dibangkitkan model; validasi `0 ≤ awal < akhir ≤ durasi`. (Diterima, sejalan dengan proposal V1 §3.5.2)
- Keluaran mentah disimpan terpisah dari hasil parsing; kegagalan parsing dicatat. (Diterima)

## Konsekuensi
- Penyimpanan AE dan rujukan segmen membutuhkan perubahan skema LeveLearn — keputusan terpisah (di luar ADR ini).

## Rujukan
- Skill `qwen35-vlm-finetune` → `references/format-data.md` §0–2a; `src/Backend-LeveLearn-main/.../prisma/schema.prisma`; proposal V1 Tabel 3.6–3.7.
