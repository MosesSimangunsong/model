# ADR-0016: Potong pita hitam dan pilih frame merata (resolusi pilot)

- **Status:** Sementara (pilot, GPU Colab T4)
- **Tanggal:** 2026-11-03
- **Pemicu tinjau ulang:** GPU lebih besar tersedia (IT Del); konfigurasi ini OOM saat training sungguhan; model salah membaca notasi pada frame; video dengan slide yang sering berganti (5 frame per segmen bisa melewatkan slide).

## Konteks
Uji 1 batch di T4 14,6 GiB (`uji_1_batch_2026-11-03_014908.json`): 768 px/14 frame (6.738 token) OOM; 512 px/14 frame (4.274 token) muat dengan puncak 13,53 GiB. Pengamatan visual sebelumnya: 512 px membuat rumus kepala tabel buram. Frame F002–F027 memiliki pita hitam (letterbox) baris 0–161 dan 918–1079 (~30% tinggi); F001 tidak. Pita sebelumnya sengaja tidak dipotong (STATUS 2 November) dengan catatan dapat ditinjau bila token terlalu banyak.

## Keputusan
1. **Pita hitam dipotong per frame** sebelum diperkecil: kotak terkecil yang memuat piksel dengan abu-abu > 30 (`cek_sampel.muat_gambar(..., potong_pita=True)`). Deteksi per frame karena tidak semua frame berpita.
2. **Frame dipilih merata** sejumlah N, selalu memuat frame pertama dan terakhir (`prompt_qg_ae.pilih_frame`). Semua `evidence_ids` pada target wajib berada di frame/transkrip yang ditampilkan (diperiksa `uji_1_batch.py`).
3. **Konfigurasi pilot: sisi terpanjang 1024 px, 5 frame** (F001, F007, F014, F021, F027) → 4.162 token untuk contoh QG. Cadangan: 768 px, 9 frame (4.210 token).
4. Batas anggaran: total token tidak melebihi 4.274 (ukuran terbesar yang sudah terbukti muat di T4).

## Alasan
Pemotongan menurunkan token per frame berpita tanpa mengubah skala isi (768 px: 336 → 240 token; 1024 px: 608 → 416). Antara lebih banyak frame bertulisan kecil (768/9) dan lebih sedikit frame bertulisan jelas (1024/5), dipilih yang kedua karena salah membaca operator (∨/∧) merusak validitas formal soal (Riset §5.1), segmen pilot berupa satu slide yang dicoret bertahap, dan urutan waktu tetap dibawa transkrip bertimestamp. Keterbacaan dinilai dari pengamatan visual frame F014, bukan dari keluaran model.

## Alternatif yang dipertimbangkan
- 512 px tanpa potong (muat, tetapi rumus buram).
- 768 px, 9 frame dengan potong (cadangan).
- Pemilihan frame berdasarkan pergantian slide — lebih tepat untuk video bervariasi, tetapi menambah komponen prapemrosesan; ditunda.
- Kernel `flash-linear-attention` untuk menghemat memori — diuji terpisah sebagai satu faktor.

## Konsekuensi
- Konfigurasi ini harus dibuktikan muat lewat uji 1 batch di Colab sebelum dipakai; anggaran 4.274 token hanya terbukti untuk forward+backward tanpa optimizer.
- Fungsi pemotongan dan pemilihan frame harus sama pada training dan inferensi; parameter (ambang 30, max_side, N) dicatat di setiap run.
- Panjang transkrip berbeda antarsegmen, sehingga anggaran frame dapat perlu disesuaikan per segmen.

## Rujukan
ADR-0006, ADR-0013; `docs/STATUS.md` (percobaan 1–3); Riset §5.1.
