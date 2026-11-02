# ADR-0005: Target modul LoRA

- **Status:** Diusulkan
- **Tanggal:** 2026-11-02
- **Pemicu keputusan:** Pilot memori dan validasi; diputuskan sebelum eksperimen utama lalu dibekukan.

## Konteks
Percobaan LoRA teks di Colab menargetkan `self_attn.q_proj`/`v_proj`. Pemeriksaan checkpoint menunjukkan konfigurasi itu hanya mengenai **8 dari 32** lapisan bahasa (lapisan full-attention), karena 24 lapisan lain memakai linear attention dengan nama modul berbeda (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`, …). MLP (`gate/up/down_proj`) tidak tersentuh. Proposal V1 §2.4.4 menyatakan modul sasaran ditentukan pada data latih/validasi.

## Opsi
- **(A)** `self_attn.q_proj`/`v_proj` saja — 8 lapisan, seperti percobaan teks.
- **(B)** Semua proyeksi linear bagian bahasa di 32 lapisan: `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `mlp.{gate,up,down}_proj`; kecualikan `visual`, `lm_head`, `embed_tokens`, `mtp`.

## Usulan
Mulai pilot multimodal dengan **(B)** karena mencakup semua lapisan; bandingkan jumlah parameter, VRAM, dan kualitas validasi terhadap (A) bila sumber daya memungkinkan. Ini **usulan**, bukan hasil.

## Konsekuensi
- Nama modul wajib diverifikasi dengan `named_modules()` pada versi Transformers yang terpasang sebelum dipakai.
- `target_modules="all-linear"` tanpa pengecualian tidak boleh dipakai (akan mengenai `visual`).
- Untuk TA: percobaan teks sebelumnya harus dilaporkan sebagai adaptasi **parsial** (8 lapisan).

## Rujukan
- Skill `qwen35-vlm-finetune` → `references/arsitektur-checkpoint.md`; proposal V1 §2.4.4 (termasuk Choi dkk. [37] tentang pemilihan modul).
