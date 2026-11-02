# ADR-0014: Aturan kerja agent AI di repo

- **Status:** Diterima
- **Tanggal:** 2026-11-02

## Keputusan
- Agent **tidak pernah** melakukan `git commit` atau `git push`; pengguna melakukannya sendiri. Agent melaporkan file yang diubah dan hasil pemeriksaan.
- Sebelum membangun sesuatu yang tidak sepele, agent menanyakan informasi yang kurang; bila banyak, dikumpulkan dalam satu file `.md` untuk dijawab.
- Agent membedakan fakta terverifikasi, keputusan, usulan, dan hal yang belum diketahui; tidak mengarang hasil.
- Agent tidak mengubah `src/` (kode LeveLearn), video, model dasar, atau data asli kecuali diminta.
- Instruksi agent terpusat di `AGENTS.md` (`CLAUDE.md` mengimpornya); skill proyek di `.claude/skills/`.

## Rujukan
- `AGENTS.md`; `Pertanyaan_Skill_VLM.md` E2–E3.
