# STATUS — serah terima antarsesi

Diperbarui: 2026-11-02. Baca file ini di awal sesi; perbarui di akhir sesi kerja (≤40 baris). Keputusan → `docs/adr/`; detail teknis → skill `qwen35-vlm-finetune`.

## Selesai (terverifikasi)
- Model `Model/Qwen3.5-9B` terunduh (4 shard). Struktur checkpoint diperiksa: 24 linear-attn + 8 full-attn; LoRA teks Colab lama hanya kena 8/32 lapisan (ADR-0005).
- Pilot BDD: `Data/Pilot/segmen_0410_0627.mp4` (137 dtk), 27 frame (`fps=1/5`, 1804×1080), transkrip Whisper `small` bahasa `id` (15 potongan).
- Format soal LeveLearn (`src/`) diperiksa: `Assessment.questions` = `[{question, options[], answer: teks, type}]`; belum ada field pembahasan/segmen.
- Chat template: format latih = format inferensi (`enable_thinking=False`) — diuji dengan render jinja.
- Infrastruktur agent: `AGENTS.md`, skill proyek, ADR 0001–0014.

- Transkrip pilot sudah dikoreksi manual (`transkrip.txt`; `transkrip.json` masih versi Whisper). Venv lokal: torch 2.15.0+cpu, transformers 5.22.1, peft 0.21.2, pillow 13.0.0.
- `scripts/cek_sampel.py` dijalankan lokal (hanya processor): token visual/frame 1904 (asli), 608 (1024), 336 (768), 160 (512); asersi gambar masuk, masking label, template latih=inferensi lolos pada target DUMMY. Keterbacaan formula per resolusi belum dicek.
- Keterbacaan frame (dilihat mata, frame 14): 1024 px jelas, 768 px terbaca tapi kecil, 512 px tidak layak → pilot pakai 1024 px (608 token/frame; 27 frame ≈ 16,4 rb token). Pita hitam letterbox tidak dipotong.
- Kontrak prompt/JSON QG-AE v0.1 diputuskan (ADR-0015, dari Riset + `Pertanyaan_Format_QG_AE.md`); `format-data.md` §2–4 diperbarui. `scripts/prompt_qg_ae.py` (prompt, susun pesan Layout A, validator, konversi LeveLearn) diuji lokal: template latih=inferensi untuk QG & AE; validator menolak contoh buruk.

- Contoh acuan pilot disetujui pengguna (pemeriksa: Moses, status `diperiksa`); `contoh_latih.jsonl` dibangun ulang (4 contoh).
- `scripts/uji_1_batch.py` ditulis (QLoRA 4-bit, LoRA pola ADR-0005 opsi B, ablasi frame asli vs kosong, catat peak VRAM). Pola LoRA diverifikasi pada struktur model (meta device): 200 modul, 32 lapisan, tanpa visual/mtp. Hanya mode `--dry` yang sudah dijalankan (lokal); **bagian GPU belum pernah dijalankan**. Colab pengguna: versi gratis (jenis GPU belum dikonfirmasi).

## Belum / berikutnya (urut)
0b. Jalankan `uji_1_batch.py` di Colab (mulai `--max-side 768 --stride 2`), catat peak VRAM / OOM / ada tidaknya kernel `fla`.
0. Contoh acuan pilot `Data/Pilot/contoh_acuan_pilot.json` (v2) sudah diperiksa & disetujui pengguna, tetapi field `annotation_status` masih `draf_ai` dan `pemeriksa` null → ubah ke `diperiksa` + isi nama pemeriksa (menunggu pengguna), lalu jalankan ulang `scripts/bangun_contoh_latih.py`.
   `scripts/bangun_contoh_latih.py` → `Data/Pilot/segmen.json` + `contoh_latih.jsonl` (1 QG + 3 AE). Diuji lewat processor: label==target, template latih=inferensi, 16.416 token gambar; total 18,5–18,7 rb token/contoh (belum diuji muat di GPU).
1. **Periksa & koreksi `Data/Pilot/transkrip.txt` secara manual** (sudah tampak salah dengar: "Knya" → kemungkinan "Q-nya").
2. Cek kualitas 27 frame (formula/tabel terbaca? ada slide terlewat?).
3. Konfirmasi jenis GPU Colab → inferensi multimodal model dasar pada segmen pilot (cek `pixel_values`, token gambar, keluaran utuh).
4. Ukur token visual per frame dengan processor; tentukan batas resolusi/jumlah frame.
5. Rancang contoh acuan QG/AE pilot (diperiksa manusia) → uji 1 batch training di Colab.

## Pertanyaan / tindakan terbuka
- Jenis GPU & langganan Colab: belum dikonfirmasi.
- Bab III revisi 1 November: akan dibuat pengguna → bandingkan dengan ADR.
- Proposal perlu diselaraskan dengan ADR-0002 (hanya QG+AE) dan ADR-0003 (QLoRA sementara).
- Perubahan skema LeveLearn untuk menyimpan AE/segmen: belum diputuskan.
