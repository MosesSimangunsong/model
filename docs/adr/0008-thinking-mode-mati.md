# ADR-0008: Mode thinking dimatikan (`enable_thinking=False`)

- **Status:** Diterima
- **Tanggal:** 2026-11-02

## Keputusan
Training dan inferensi memakai `enable_thinking=False`, sama dengan percobaan Colab sebelumnya. Target latih berisi jawaban langsung tanpa jejak penalaran tersembunyi; AE tetap memuat pembahasan lengkap (langkah, alasan kunci, alasan tiap pengecoh).

## Fakta terverifikasi
Merender `chat_template.jinja` asli (2 November 2026): render percakapan latih = prompt generasi (`enable_thinking=False`) + target + `<|im_end|>\n`. Bentuk latih dan inferensi konsisten; scaffold `<think>\n\n</think>\n\n` berada di prompt.

## Konsekuensi
- Penalaran yang ingin dinilai harus tertulis eksplisit di pembahasan AE, bukan di blok `<think>`.

## Rujukan
- `Pertanyaan_Skill_VLM.md` C4; `Model/Qwen3.5-9B/chat_template.jinja`.
