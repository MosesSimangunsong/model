# Progres 2–3 November 2026 — Persiapan Alur Fine-Tuning Qwen3.5-9B (QG + AE) LeveLearn

Dokumen ini mencatat seluruh pekerjaan dari awal sampai uji GPU terakhir, termasuk alasan setiap langkah, angka hasil, masalah yang muncul dan cara penyelesaiannya, serta istilah teknis yang dipakai. Pekerjaan dimulai 2 November dan berlanjut sampai 3 November 2026; keduanya dicatat di sini.

Aturan baca:
- **Terverifikasi** = sudah dijalankan dan keluarannya dilihat.
- **Perkiraan/dugaan** = belum diukur; ditandai eksplisit.
- Angka uji GPU berasal dari berkas JSON yang dihasilkan script di Colab dan dikirim pengguna.

---

## Daftar isi

1. [Ringkasan singkat](#1-ringkasan-singkat)
2. [Tujuan dan batasan proyek](#2-tujuan-dan-batasan-proyek)
3. [Glosarium istilah teknis](#3-glosarium-istilah-teknis)
4. [Lingkungan dan alat](#4-lingkungan-dan-alat)
5. [Gambaran alur teknis](#5-gambaran-alur-teknis)
6. [Kronologi pekerjaan rinci](#6-kronologi-pekerjaan-rinci)
7. [Hasil uji GPU (percobaan 1–4)](#7-hasil-uji-gpu-percobaan-14)
8. [Tabel angka penting](#8-tabel-angka-penting)
9. [Keputusan yang tercatat (ADR)](#9-keputusan-yang-tercatat-adr)
10. [Masalah, penyebab, dan penyelesaian](#10-masalah-penyebab-dan-penyelesaian)
11. [Berkas yang dihasilkan dan fungsinya](#11-berkas-yang-dihasilkan-dan-fungsinya)
12. [Batasan: yang belum terbukti](#12-batasan-yang-belum-terbukti)
13. [Langkah berikutnya](#13-langkah-berikutnya)

---

## 1. Ringkasan singkat

Dalam dua hari ini, alur persiapan fine-tuning dibangun dan diuji **untuk satu segmen pilot** (video Logika Proposisi, topik tabel kebenaran, 137 detik):

1. Bahan masukan disiapkan: 27 frame gambar dan transkrip ucapan (dikoreksi manual).
2. Cara gambar diubah menjadi token diukur, dan resolusi dipilih berdasarkan keterbacaan.
3. Format prompt dan keluaran JSON untuk QG dan AE diputuskan berdasarkan riset (ADR-0015).
4. Contoh jawaban acuan (3 soal + 3 pembahasan) disusun AI, ditinjau, direvisi, lalu **diperiksa dan disetujui manusia** (pemeriksa: Moses).
5. Contoh acuan dipecah menjadi 4 contoh latih (1 QG + 3 AE).
6. Script uji 1 batch dijalankan di Google Colab (GPU Tesla T4). Setelah empat percobaan dan beberapa perbaikan, **konfigurasi pilot terbukti muat**: 1024 px, 5 frame, pita hitam dipotong, 4.162 token, puncak memori GPU 13,28 GiB dari 14,6 GiB.

Yang **belum** dilakukan: training sungguhan (langkah optimizer), menyimpan dan memuat adapter, menyuruh model menulis soal, dan membandingkan model dasar dengan model hasil latih.

---

## 2. Tujuan dan batasan proyek

### 2.1 Tujuan
Membangkitkan, dari segmen video pembelajaran Logika Proposisi:
- **QG (Question Generation)** — 1 pemanggilan model per segmen → 3 soal pilihan ganda, masing-masing 4 opsi (A–D), tepat 1 kunci.
- **AE (Answer Explanation)** — 1 pemanggilan per soal → pembahasan: aturan yang dipakai, langkah penyelesaian, alasan kunci benar, alasan setiap pengecoh salah, dan rujukan bukti.

Masukan model: **frame terurut + transkrip ASR + instruksi** (ADR-0006). Model: `Qwen/Qwen3.5-9B` versi post-trained (ADR-0001), dilatih dengan QLoRA untuk pilot (ADR-0003).

### 2.2 Batasan yang dipegang selama bekerja
- Model sudah ditetapkan; tidak diganti.
- Keluaran model hanya QG dan AE (ADR-0002).
- Keluaran AI yang belum diperiksa manusia bukan data benar.
- Loss pada training harus hanya dihitung di bagian jawaban, dan gambar harus benar-benar masuk ke model.
- Penurunan loss bukan bukti kualitas.
- Agent tidak melakukan `git commit` atau `git push`; semua dilakukan pengguna.

---

## 3. Glosarium istilah teknis

Istilah diurutkan menurut kemunculannya dalam alur.

| Istilah | Arti sederhana | Kaitan dengan proyek |
|---|---|---|
| **VLM (Vision-Language Model)** | Model AI yang menerima gambar dan teks sekaligus | Qwen3.5-9B adalah VLM; ia melihat frame dan membaca transkrip |
| **Parameter** | Angka-angka di dalam model yang menentukan cara model "berpikir" | "9B" = sekitar 9 miliar parameter (tepatnya 9.449.921.776 setelah dimuat) |
| **Checkpoint** | Berkas bobot model yang tersimpan | `Model/Qwen3.5-9B`, 4 berkas `.safetensors` (sekitar 19 GB) |
| **Segmen** | Potongan video | Pilot: detik 250–387 dari video penuh |
| **Frame** | Satu gambar diam yang diambil dari video | 27 frame, satu tiap 5 detik |
| **ASR (Automatic Speech Recognition)** | Mengubah ucapan menjadi teks | Whisper; model tidak bisa mendengar audio |
| **Whisper / faster-whisper** | Model ASR dari OpenAI; faster-whisper adalah versi yang lebih ringan | Dipakai ukuran `small`, di CPU |
| **VAD (Voice Activity Detection)** | Mendeteksi bagian yang berisi suara | `vad_filter=True` membuang bagian hening |
| **FFmpeg** | Alat baris perintah untuk memotong video dan mengambil frame | Memotong segmen dan mengambil frame `fps=1/5` |
| **Token** | Potongan kecil teks atau gambar yang dibaca model dalam bentuk angka | Panjang masukan dan memori dihitung dalam token |
| **Token visual** | Token yang mewakili sepetak gambar | Satu token visual = area 32 × 32 piksel |
| **Processor** | Alat yang mengubah gambar dan teks menjadi token | `Qwen3VLProcessor` |
| **Chat template** | Aturan baku cara menyusun percakapan menjadi teks untuk model | Dari berkas `chat_template.jinja` di checkpoint |
| **`enable_thinking=False`** | Mematikan mode "berpikir panjang" model | Training dan inferensi sama-sama mati (ADR-0008) |
| **Prompt** | Masukan untuk model (instruksi + data) | Pesan `system` + `user` |
| **Target** | Jawaban benar yang ingin ditiru model | JSON QG atau AE dari contoh acuan |
| **Inferensi** | Menyuruh model menjawab tanpa mengubah bobotnya | Belum dikerjakan untuk model 9B |
| **Training / fine-tuning** | Menyesuaikan bobot model dengan contoh | Belum dikerjakan; baru uji 1 batch |
| **Forward (maju)** | Model membaca masukan dan menghasilkan tebakan | Bagian pertama uji 1 batch |
| **Loss** | Angka yang mengukur seberapa jauh tebakan model dari jawaban benar; makin kecil makin dekat | Loss model dasar pada contoh pilot ≈ 0,67 |
| **Cross-entropy** | Rumus loss standar untuk menebak token berikutnya | Dipakai untuk menghitung loss |
| **Backward (mundur)** | Menghitung arah perubahan bobot dari loss | Bagian kedua uji 1 batch |
| **Gradien** | Hasil backward: arah dan besar perubahan tiap parameter | Diperiksa tidak kosong dan tidak NaN |
| **Optimizer / optimizer step** | Algoritma yang benar-benar mengubah bobot memakai gradien | **Belum dijalankan** |
| **Label `-100`** | Penanda posisi yang tidak dihitung dalam loss | Seluruh prompt dan gambar diberi `-100`; hanya jawaban yang dinilai |
| **Logits** | Skor mentah model untuk setiap kata di kosakata, di setiap posisi | Kosakata 248.320 kata; sangat besar di memori |
| **`logits_to_keep`** | Opsi model untuk menghitung logits hanya di sejumlah posisi terakhir | Dipakai agar logits hanya dihitung di bagian jawaban |
| **LoRA (Low-Rank Adaptation)** | Menambahkan lapisan kecil di samping lapisan asli dan hanya melatih lapisan kecil itu | 40,1 juta parameter dilatih (0,42%) |
| **Adapter** | Hasil LoRA yang disimpan terpisah dari model dasar | Akan dihasilkan oleh training |
| **r, alpha, dropout (LoRA)** | Ukuran lapisan kecil (r), skala pengaruhnya (alpha), dan peluang mematikan sebagian saat latih (dropout) | r=16, alpha=32, dropout=0,05 untuk uji (bukan nilai final) |
| **Kuantisasi** | Menyimpan angka dengan bit lebih sedikit agar hemat memori | Model dimampatkan ke 4-bit |
| **NF4, double quantization** | Jenis kuantisasi 4-bit (NormalFloat-4) dan teknik memampatkan konstanta kuantisasinya juga | Pengaturan `BitsAndBytesConfig` |
| **QLoRA** | LoRA di atas model yang dikuantisasi 4-bit | Metode pilot (ADR-0003) |
| **bitsandbytes** | Pustaka yang melakukan kuantisasi 4-bit di GPU | Versi 0.50.2 di Colab |
| **VRAM** | Memori di GPU | T4: 14,6 GiB tersedia |
| **OOM (Out Of Memory)** | Kehabisan memori GPU | Terjadi pada 768 px, 14 frame |
| **GiB** | Gibibyte = 1.073.741.824 byte (sedikit lebih besar dari GB) | Satuan VRAM di catatan |
| **fp16 / bf16** | Format angka 16-bit. bf16 lebih stabil, tetapi hanya didukung native di GPU baru | T4 memakai fp16 |
| **Compute capability** | Generasi kemampuan GPU NVIDIA | T4 = 7.5; bf16 native butuh ≥ 8.0 |
| **Gradient checkpointing** | Menghemat memori dengan menghitung ulang sebagian nilai saat backward, bukan menyimpannya | Aktif (`use_reentrant=False`) |
| **SDPA** | Implementasi attention bawaan PyTorch | `attn_implementation="sdpa"` |
| **Full attention** | Setiap token membandingkan dirinya dengan semua token sebelumnya | 8 dari 32 lapisan Qwen3.5 |
| **Linear attention** | Model menyimpan ringkasan berjalan, tidak membandingkan semua token | 24 dari 32 lapisan Qwen3.5 |
| **Kernel** | Program khusus untuk GPU yang mengerjakan satu operasi secara efisien | `flash-linear-attention`, `causal-conv1d` belum terpasang |
| **Fallback** | Cara cadangan yang dipakai bila kernel khusus tidak ada | Benar tetapi lambat dan kemungkinan boros memori |
| **CUDA / cuDNN** | CUDA = platform pemrograman GPU NVIDIA; cuDNN = pustaka NVIDIA untuk operasi jaringan saraf, termasuk konvolusi | Sumber error di percobaan 2 |
| **conv3d / patch_embed** | Lapisan awal pembaca gambar yang memotong gambar menjadi petak | Tempat error cuDNN terjadi |
| **Vision encoder** | Bagian model yang membaca gambar | Dibekukan, tidak dilatih (ADR-0004) |
| **mtp** | Modul *multi-token prediction* di checkpoint | Tidak dimuat oleh kelas model; tidak dilatih |
| **Meta device** | Membuat struktur model tanpa memuat bobot | Dipakai di laptop untuk memeriksa nama modul |
| **Regex `target_modules`** | Pola teks untuk memilih modul yang dipasangi LoRA | Pola opsi (B) ADR-0005 |
| **Ablasi** | Membandingkan hasil dengan satu bagian dihilangkan | Frame asli vs frame abu-abu kosong |
| **Letterbox (pita hitam)** | Blok hitam di atas dan bawah gambar | Dipotong (ADR-0016) |
| **JSON / JSONL** | Format data terstruktur; JSONL = satu objek JSON per baris | Keluaran model dan berkas contoh latih |
| **Manifest** | Berkas daftar isi segmen | `segmen.json` |
| **`evidence_ids`** | ID frame (`F001`…) atau potongan transkrip (`T001`…) yang mendukung soal | Validator memeriksa ID ada |
| **Validator** | Kode pemeriksa format keluaran | `validasi_qg`, `validasi_ae` |
| **Split train/val/test** | Pembagian data untuk latih, validasi, dan uji | Per sumber video (ADR-0011); belum dilakukan |
| **`source_group_id`** | ID kelompok sumber agar semua contoh dari satu video berada di split yang sama | `pilot_bdd_source_01` |
| **ADR (Architecture Decision Record)** | Catatan keputusan beserta alasan | `docs/adr/`, 16 ADR |
| **Epoch** | Satu putaran melewati seluruh data latih | Dipakai untuk perkiraan waktu training |

---

## 4. Lingkungan dan alat

### 4.1 Laptop (terverifikasi)
| Komponen | Nilai |
|---|---|
| OS | Windows 11, PowerShell |
| GPU | RTX 3060 Laptop 6 GB — tidak cukup untuk model 9B (4-bit saja ±7,4 GiB) |
| Python venv | `.venv` |
| Paket | torch 2.15.0+cpu, transformers 5.22.1, peft 0.21.2, accelerate 1.15.0, pillow 13.0.0 |
| Alat lain | FFmpeg (winget `Gyan.FFmpeg`), faster-whisper (CPU, karena cuBLAS CUDA tidak ada) |
| Model lokal | `Model/Qwen3.5-9B` (di disk tertulis huruf kecil `model/`) |

Laptop dipakai untuk prapemrosesan, anotasi, menulis kode, dan **memeriksa token tanpa memuat bobot model**.

### 4.2 Google Colab (terverifikasi dari keluaran pengguna)
| Komponen | Nilai |
|---|---|
| Langganan | Gratis |
| GPU | Tesla T4, 14,6 GiB, compute capability 7.5 |
| Lingkungan bersih (percobaan 3–4) | torch 2.11.0+cu130, CUDA 13.0, cuDNN 9.27 (dilaporkan `92700`), transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 |
| Kernel cepat | `flash-linear-attention` dan `causal-conv1d` tidak terpasang |
| Lokasi model | Google Drive: `MyDrive/LeveLearn/models/Qwen3.5-9B` |

Versi transformers di Colab sama dengan laptop (5.22.1), sehingga pemeriksaan processor, template, dan nama modul di laptop berlaku juga di Colab.

---

## 5. Gambaran alur teknis

```
VIDEO SUMBER (Data/Video, tidak diubah)
   │ FFmpeg: potong 250–387 dtk (libx264 CRF 18)
   ▼
SEGMEN PILOT (segmen_0410_0627.mp4, 137 dtk)
   │                                   │
   │ FFmpeg fps=1/5                    │ faster-whisper small, CPU, VAD
   ▼                                   ▼
27 FRAME (1804×1080)              TRANSKRIP (15 potongan) → koreksi manual
   │                                   │
   └──────────────┬────────────────────┘
                  ▼
   MANIFEST segmen.json (ID F001–F027, T001–T015, waktu relatif & absolut)
                  │
   CONTOH ACUAN (3 QG + 3 AE) ← disusun AI → ditinjau → direvisi → diperiksa manusia
                  │ bangun_contoh_latih.py
                  ▼
   contoh_latih.jsonl: 1 contoh QG + 3 contoh AE
                  │
   PENYUSUNAN MASUKAN (prompt_qg_ae.susun_pesan, Layout A)
   [system] aturan tetap
   [user]   metadata → frame berlabel ID+waktu (gambar) → transkrip berlabel → tugas
                  │ processor Qwen3VL + chat template (enable_thinking=False)
                  ▼
   TOKEN: input_ids, pixel_values, image_grid_thw, labels (-100 di prompt)
                  │
   MODEL 9B 4-bit (NF4) + LoRA (40,1 jt parameter, 32 lapisan bahasa)
                  │ forward → loss (hanya token jawaban) → backward → gradien
                  ▼
   ✅ UJI 1 BATCH (tanpa optimizer step)        ⬜ training → adapter → generasi → validasi → evaluasi
```

---

## 6. Kronologi pekerjaan rinci

### 6.1 Fondasi (dikerjakan sebelum percakapan ini; sumber: `docs/STATUS.md` dan ADR)
- **Model diunduh** ke `Model/Qwen3.5-9B` (4 shard `.safetensors`).
- **Struktur checkpoint diperiksa** tanpa memuat bobot (`references/arsitektur-checkpoint.md`):
  - Kelas `Qwen3_5ForConditionalGeneration`; processor `Qwen3VLProcessor`.
  - Bagian bahasa 32 lapisan, **hibrida**: pola 3 lapisan linear attention lalu 1 lapisan full attention. Full attention hanya di lapisan 3, 7, 11, 15, 19, 23, 27, 31.
  - Akibat penting: LoRA teks di notebook Colab lama (target `q_proj`/`v_proj`, r=4) hanya mengenai **8 dari 32 lapisan**. Angkanya cocok persis: 69.632 parameter/lapisan × 8 = 557.056, sama dengan yang dilaporkan notebook. Karena itu LoRA lama harus dilaporkan sebagai adaptasi parsial (ADR-0005).
  - Vision encoder: 27 blok, patch 16, spatial merge 2. Token khusus gambar `<|image_pad|>` = ID 248056.
- **Chat template diuji** dengan merender `chat_template.jinja` asli: render percakapan penuh = prompt (dengan `add_generation_prompt=True`, `enable_thinking=False`) + target + `<|im_end|>\n`. Artinya format latih dan format inferensi sama.
- **Format soal LeveLearn diperiksa** di `src/`: `Assessment.questions` berisi `[{question, options[], answer, type}]`, dengan `answer` berupa teks opsi. Belum ada tempat untuk pembahasan atau segmen.
- **Infrastruktur kerja**: `AGENTS.md`, skill proyek `qwen35-vlm-finetune`, dan ADR 0001–0014.

### 6.2 Prapemrosesan segmen pilot (ADR-0013)
| Langkah | Alat dan parameter | Hasil |
|---|---|---|
| Potong segmen | FFmpeg, re-encode libx264 CRF 18 agar titik potong akurat | `segmen_0410_0627.mp4`, 250–387 dtk absolut (137 dtk) |
| Ambil frame | FFmpeg `fps=1/5`, JPEG `-q:v 2` | 27 frame 1804×1080 |
| Waktu frame | relatif = (N−1)×5; absolut = relatif + 250 | F001 = 250 dtk … F027 = 380 dtk |
| ASR | `scripts/transkrip.py`: faster-whisper `small`, `compute_type=int8`, CPU, `vad_filter=True` | 15 potongan, bahasa terdeteksi `id` |

Isi segmen (dilihat langsung dari frame): slide "Motivation Through Truth Tables" yang menampilkan tabel kebenaran 8 baris untuk `p ∨ (q ∧ r)` dan versi ringkasnya 4 baris dengan tanda `*`. Pengajar menggabungkan baris 1–4 menjadi `T * * T`, lalu menjelaskan baris 5 dan 6. Baris 7–8 belum dijelaskan karena segmen berhenti di "Sekarang yang terakhir".

### 6.3 Koreksi transkrip
- Whisper salah mendengar beberapa istilah, misalnya "Knya" yang seharusnya "Q nya", dan "menguakili" yang seharusnya "mewakili".
- Pengguna mengoreksi `transkrip.txt` secara manual (dua putaran; baris 8 dan 11 diperbaiki pada putaran kedua). Kata "eh sorry" dibiarkan karena memang diucapkan pembicara.
- `transkrip.json` **tidak** dikoreksi dan masih keluaran mentah Whisper. Script memakai `transkrip.txt`.

### 6.4 Pemasangan pustaka di laptop
Dipasang `torch` dan `torchvision` versi CPU, `transformers`, `peft`, dan `pillow`. Tujuannya memuat **processor** (bukan model) agar penyusunan token bisa diuji tanpa GPU.

### 6.5 Uji penyusunan sampel di laptop (`scripts/cek_sampel.py`)
Yang diperiksa:
1. **Token visual per frame** pada beberapa resolusi (hasil di §8.1). Perkiraan awal ADR-0006 (~1.900 token/frame) terkonfirmasi: 1.904 token.
2. **Gambar benar-benar masuk**: jumlah token `<|image_pad|>` di `input_ids` sama dengan `image_grid_thw / merge²`.
3. **Masking label**: semua label prompt dan gambar bernilai `-100`; token yang dinilai hanya jawaban.
4. **Template latih = inferensi**: hasilnya `True`.

Cara membangun satu contoh: prompt ditokenisasi bersama gambar oleh processor, jawaban ditokenisasi terpisah lalu disambung di ujung. Label prompt diisi `-100` dan label jawaban = token jawaban. Pada tahap ini target masih **dummy** (hanya menguji mekanik).

### 6.6 Keterbacaan resolusi
Frame 14 (tabel dengan coretan pengajar) dilihat pada 1024, 768, dan 512 px:
- 1024 px: jelas; rumus `p ∨ (q ∧ r)` dan nilai T/F terbaca.
- 768 px: terbaca tetapi kecil; coretan menimpa angka.
- 512 px: tidak layak; rumus kepala tabel buram.

Juga ditemukan **pita hitam (letterbox)** di atas dan bawah frame F002–F027 (sekitar 30% tinggi). Saat itu diputuskan tidak dipotong dulu, dengan catatan bisa ditinjau bila token terlalu banyak. Belakangan ini ditinjau lagi (§6.14).

### 6.7 Riset format dan keputusan kontrak QG/AE (ADR-0015)
Sumber: `docs/research/Riset_Prompt_QG_AE_dan_Masukan_VLM_LeveLearn.md` (12 rujukan, misalnya QuizTools, QGEval, Moore dkk. 2023, Haladyna 2022, Yu dkk. AAAI 2026).

Riset dibandingkan dengan rancangan lama di skill. Tujuh perbedaan dikumpulkan di `docs/pertanyaan/Pertanyaan_Format_QG_AE.md` dan dijawab pengguna:

| No | Keputusan |
|---|---|
| P1 | Field `concept` (konsep yang diuji), bukan `learning_objective` |
| P2 | Field `stimulus` terpisah dari `stem`; saat konversi ke LeveLearn digabung menjadi `question`. Stimulus harus teks yang terbaca siswa |
| P3 | `status` + `issue`: QG ∈ {ok, insufficient_context}; AE ∈ {ok, insufficient_context, invalid_question}. Validator tetap memeriksa sendiri |
| P4 | `evidence_ids`: `F001…` untuk frame, `T001…` untuk transkrip |
| P5 | AE: `principle`, `steps[]`, `conclusion`, `distractor_explanations[]` (tepat 3 opsi selain kunci), `evidence_ids`. Tanpa pengayaan |
| P6 | Waktu di masukan memakai detik absolut video (250–387) dengan label konvensi |
| P7 | Contoh acuan disusun AI sebagai draf, diperiksa manusia |

Juga diuji bahwa pesan dengan peran `system` didukung template checkpoint dan tidak merusak kesamaan latih = inferensi.

### 6.8 Modul prompt dan validator (`scripts/prompt_qg_ae.py`)
Satu sumber yang sama untuk training dan inferensi:
- `SYSTEM`, `INSTRUKSI_QG`, `INSTRUKSI_AE` — teks prompt (adaptasi Riset §7).
- `baca_transkrip` — membaca `transkrip.txt` menjadi potongan berlabel `T001…` beserta waktu absolut.
- `pilih_frame` — memilih N frame merata yang selalu memuat frame pertama dan terakhir (ditambahkan 3 Nov, ADR-0016).
- `susun_pesan` — **Layout A**: `[system]` aturan → `[user]` metadata segmen → frame berlabel `Frame F00X | detik` diikuti gambarnya → transkrip berlabel → tugas (QG, atau AE + soal JSON).
- `validasi_qg`, `validasi_ae` — memeriksa status, jumlah soal, opsi A–D unik, kunci valid, field terisi, pengecoh tepat 3, dan `evidence_ids` ada di segmen.
- `ke_levelearn` — mengubah satu soal ke format `Assessment.questions`.

Diuji di laptop: contoh benar lolos; contoh buruk ditolak dengan alasannya (opsi duplikat, ID `F099` tidak ada, jumlah soal salah, pengecoh kurang).

### 6.9 Contoh acuan pilot (`Data/Pilot/contoh_acuan_pilot.json`)
**Draf v1** disusun dari frame aktual dan transkrip terkoreksi, lalu dicek dengan validator dan **enumerasi semua kombinasi T/F** (bukan hanya keyakinan).

**Tinjauan** (`docs/tinjauan/response_json.md`) menemukan masalah. Setiap poin diverifikasi:
- Kunci selalu opsi terpanjang (68/71/74 karakter) → benar, diperbaiki.
- Pengecoh q3 mudah dicoret hanya dengan membaca tabel → benar, diganti pengecoh berbasis kesalahan penalaran.
- q1 dan q2 tumpang tindih → benar, q2 diganti total.
- Aturan penggabungan kurang lengkap → ditambahkan ("semua kombinasi yang diwakili `*` harus menghasilkan hasil sama").
- Klaim "pengajar menggabungkan bertahap" dianggap tak terverifikasi → sebenarnya didukung transkrip T004–T011, tetapi kalimat itu hilang karena q2 diganti.

**Draf v2** (disetujui):

| Soal | Konsep | Kunci | Bukti |
|---|---|---|---|
| q1 | Makna tanda `*` pada tabel ringkas | A | F001, F027, T006 |
| q2 | Baris ringkas yang mewakili tepat baris 1–2 (`p=T, q=T, r=*`) | D | F001, T004–T006 |
| q3 | Syarat sah penggabungan baris 5–6 (hasil berbeda mengikuti r) | B | F001, T013, T014 |

Pemeriksaan v2: validator lolos; enumerasi membuktikan hanya opsi D yang mewakili tepat baris 1–2 dan penggabungan baris 5–6 tidak sah; kunci tidak lagi terpanjang. Pengguna memeriksa dan menyetujui; status diubah menjadi `diperiksa`, pemeriksa `Moses`.

### 6.10 Pemecahan menjadi contoh latih (`scripts/bangun_contoh_latih.py`)
Menghasilkan:
- `Data/Pilot/segmen.json` — manifest (27 frame dan 15 potongan transkrip dengan ID dan waktu).
- `Data/Pilot/contoh_latih.jsonl` — 4 baris: 1 QG (konteks → 3 soal) dan 3 AE (konteks + 1 soal → pembahasan). Semua memakai `source_group_id` yang sama agar nantinya tetap di split yang sama.

Script memvalidasi target sebelum menulis berkas. Pemeriksaan lewat processor dengan target asli (27 frame, 1024 px, tanpa potong pita):

| Contoh | Total token | Token gambar | Token dinilai | Label = target | Template sama |
|---|---|---|---|---|---|
| QG | 18.658 | 16.416 | 707 | ya | ya |
| AE q1 | 18.550 | 16.416 | 418 | ya | ya |
| AE q2 | 18.595 | 16.416 | 437 | ya | ya |
| AE q3 | 18.622 | 16.416 | 443 | ya | ya |

Contoh QG adalah yang terpanjang, sehingga menjadi uji kasus terberat.

### 6.11 Script uji 1 batch (`scripts/uji_1_batch.py`)
Tujuan: membuktikan alur training bekerja **sebelum** training panjang. Tidak ada langkah optimizer, jadi bobot tidak berubah.

Isi script (versi akhir):
1. Memuat processor, membangun satu contoh dari JSONL dengan frame dipilih merata dan pita dipotong.
2. Memastikan target berstatus `diperiksa` dan semua `evidence_ids` ada di frame atau transkrip yang ditampilkan.
3. Memilih dtype: bf16 hanya bila compute capability ≥ 8.0; selain itu fp16.
4. Memuat model 4-bit: `BitsAndBytesConfig(load_in_4bit, nf4, double quant, compute dtype)`, `attn_implementation="sdpa"`, `device_map={"": 0}`.
5. Mengaktifkan gradient checkpointing (`use_reentrant=False`) dan `enable_input_require_grads()`. `prepare_model_for_kbit_training` **tidak** dipakai secara bawaan karena ia mengubah sebagian parameter ke fp32 yang memakan beberapa GB di T4 (tersedia lewat `--kbit-prep`).
6. Memasang LoRA dengan pola regex opsi (B):
   ```
   model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)
   ```
   Pemeriksaan wajib: semua parameter yang dilatih adalah LoRA, tidak ada di `visual` atau `mtp`, dan mencakup 32 lapisan.
7. Forward dengan **logits hemat**: logits hanya dihitung untuk `token_label + 1` posisi terakhir (`logits_to_keep`), lalu cross-entropy dihitung sendiri. Ini setara dengan loss bawaan karena label hanya ada di ujung urutan, dan script memeriksa syarat itu.
8. Backward: memeriksa gradien tidak kosong dan finite; mencatat waktu dan puncak VRAM.
9. Ablasi: loss dihitung ulang dengan frame diganti abu-abu polos berukuran sama.
10. `--sweep`: beberapa konfigurasi `max_side:jumlah_frame` dicoba berurutan **tanpa memuat ulang model**; berhenti di konfigurasi pertama yang berhasil. OOM dicatat beserta fasenya (maju, mundur, atau ablasi).
11. Semua parameter, versi paket, GPU, dan hasil disimpan ke `runs/uji_1_batch_<waktu>.json`.

Pola LoRA diverifikasi di laptop dengan membuat model di **meta device**: 200 modul cocok di 32 lapisan, tidak ada modul visual, dan kelas model tidak memuat modul `mtp`.

### 6.12 Kerapian repo dan git
- Usulan struktur rapi disusun. Pengguna memindahkan `Pertanyaan_Format_QG_AE.md` ke `docs/pertanyaan/`, `response_json.md` ke `docs/tinjauan/`, dan notebook lama ke `notebooks/`.
- `.gitignore` diperluas: lingkungan Python, `Model/`, `Data/Video/`, `*.mp4`, artefak training besar (`*.safetensors`, `*.bin`, `*.pt`, adapter dan checkpoint di `runs/`), berkas rahasia (`.env`), salinan aplikasi `src/` (berisi database `your_database_file.db` dan unggahan 57 MB), lalu oleh pengguna juga `data/` dan `docs/proposal/`. Berkas yang terdeteksi git turun dari 828 menjadi 81.
- Branch: `feat/persiapan-finetuning-qwen35-pilot`. Pengguna membuat 14 commit bertopik (aturan git, instruksi agent, konteks, ADR, skill, prapemrosesan, riset, script, contoh acuan, uji 1 batch, notebook, STATUS, dan pengabaian data).
- Remote: `origin` → `github.com/MosesSimangunsong/model.git`. Perintah push diberikan; apakah sudah dijalankan tidak diverifikasi dari sini.
- Catatan: folder di disk berhuruf kecil (`data`, `model`, `pilot`, `video`, `proposal`), sementara kode memakai `Data/Pilot` dan `Model/`. Di Windows tidak masalah, tetapi di Colab (Linux) nama folder harus dibetulkan setelah unzip. Karena `data/` diabaikan git, data pilot dibawa ke Colab lewat zip.

### 6.13 Persiapan Colab
- Isi zip `pilot_colab.zip` (sekitar 1,5 MB): folder `scripts/`, 27 frame, `segmen.json`, `contoh_latih.jsonl`, `transkrip.txt`.
- Model dibaca langsung dari Drive. Folder awalnya tidak ditemukan karena ejaan: yang benar `LeveLearn`, bukan `Levelearn`. Sel diagnostik (`os.listdir`, `find -iname "*qwen*"`) menemukan path yang benar.
- Jatah GPU gratis sempat habis (Colab tidak memberi GPU; `nvidia-smi: command not found`), sehingga uji ditunda.

### 6.14 Potong pita hitam dan pemilihan frame (ADR-0016)
Dipicu oleh hasil percobaan 3: yang muat (512 px) kurang terbaca, sedangkan 768 px tidak muat.
- Pemeriksaan per frame: F002–F027 berisi konten di baris 162–917; F001 tidak berpita. Pemotongan harus per frame.
- Pemotongan: kotak terkecil berisi piksel dengan abu-abu > 30, sebelum gambar diperkecil. Skala isi tidak berubah; hanya area hitam yang dibuang.
- Pemilihan frame merata yang selalu memuat frame awal dan akhir, karena jawaban acuan merujuk F001 dan F027. Script menolak konfigurasi yang membuang frame bukti (diuji dengan membuang F027).
- Anggaran: total token tidak melebihi 4.274 (ukuran terbesar yang terbukti muat). Dua kandidat: 768 px × 9 frame (4.210) dan 1024 px × 5 frame (4.162).
- Dipilih **1024 px × 5 frame (F001, F007, F014, F021, F027)** karena salah membaca ∨/∧ merusak validitas soal, frame segmen ini banyak yang hampir sama, dan urutan waktu dibawa transkrip.

---

## 7. Hasil uji GPU (percobaan 1–4)

Semua di Colab, Tesla T4 14,6 GiB, contoh QG (yang terpanjang), LoRA r=16/alpha=32, seed 42.

### Percobaan 1 — 2 Nov (OOM, dengan catatan penting)
- dtype **bf16** (salah pilih: torch melaporkan bf16 didukung di T4 lewat emulasi yang lambat).
- Memuat model dari Drive: 8 menit 4 detik. VRAM setelah muat: **7,38 GiB**.
- LoRA: **40.108.032 parameter dilatih dari 9.449.921.776 (0,4244%)**, lapisan 0–31.
- Peringatan: kernel `causal_conv1d` dan `chunk_gated_delta_rule` (flash-linear-attention) tidak terpasang, sehingga memakai implementasi PyTorch yang "benar tetapi jauh lebih lambat".
- **OOM** pada 768 px × 14 frame (6.738 token). Fase OOM tidak tercatat.
- Perbaikan sesudahnya: fp16 untuk GPU < 8.0, logits hemat, `--sweep`, pencatatan fase OOM.

### Percobaan 2 — 3 Nov (error cuDNN)
- fp16 terpilih dengan benar; muat 4 menit 20 detik; VRAM 7,38 GiB; LoRA sama.
- Gagal di `patch_embed` (conv3d) dengan `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`. Ini masalah lingkungan, bukan OOM.
- Diagnosis tanpa memuat model:
  - Sel A: torch yang berjalan 2.11.0+cu130 (cuDNN 9.20), tetapi pip mencatat torch 2.12.1, torchvision 0.27.1, dan dua paket cuDNN (cu12 9.27.0.42 dan cu13 9.20.0.48).
  - Sel B: conv3d kecil **gagal dengan cuDNN** dan **berhasil tanpa cuDNN**.
- Dugaan penyebab: `pip install -U ... torchvision` ikut meng-upgrade torch dan mencampur dua versi CUDA. Ditambahkan opsi `--sin-cudnn`.

### Percobaan 3 — 3 Nov (berhasil di 512 px)
Runtime baru dan pip **tanpa** torchvision → torch tetap 2.11.0+cu130, cuDNN aktif tanpa error. Ini mendukung dugaan pada percobaan 2.

| Konfigurasi | Token (gambar) | Hasil | Puncak VRAM |
|---|---|---|---|
| 768 px × 14 frame | 6.738 (4.704) | OOM saat **maju** | 11,64 GiB saat gagal |
| 512 px × 14 frame | 4.274 (2.240) | **OK** | 13,53 GiB |

512 px: loss 0,6659; 27,8 detik maju+mundur; ablasi frame asli 0,6659 < frame kosong 0,7049. Muat model 400 detik.

### Percobaan 4 — 3 Nov (konfigurasi pilot terbukti muat)
| Butir | Nilai |
|---|---|
| Lingkungan | torch 2.11.0+cu130, CUDA 13.0, cuDNN aktif (`92700`), tanpa `--sin-cudnn` |
| Konfigurasi | 1024 px × 5 frame (F001, F007, F014, F021, F027), pita dipotong |
| Token | 4.162 total (gambar 2.272, dinilai 707) |
| Hasil | **OK** |
| Puncak VRAM | **13,28 GiB** (sisa ±1,3 GiB) |
| Waktu maju+mundur | 29,4 detik |
| Loss | 0,6722 |
| Ablasi | frame asli 0,6722 < frame kosong 0,7086 |
| Muat model | 85 detik (bervariasi antarsesi) |

### Cara membaca hasil uji
- **Loss 0,67 adalah loss model dasar.** LoRA dimulai dengan matriks B bernilai nol dan belum ada optimizer step, sehingga adapter belum berpengaruh. Itu juga menjelaskan mengapa loss train-mode dan eval-mode sama persis.
- **Loss percobaan 3 dan 4 tidak bisa dibandingkan** sebagai kualitas, karena masukannya berbeda.
- **Ablasi** menunjukkan model memakai gambar: loss lebih rendah dengan frame asli. Ini pemeriksaan kewajaran pada satu contoh, bukan evaluasi.
- **Sisa memori sekitar 1,3 GiB** belum tentu cukup untuk training sungguhan, karena optimizer AdamW menambah memori. Perkiraan kasar 0,3–0,5 GiB untuk 40 juta parameter; belum diukur.
- **Perkiraan waktu (ekstrapolasi kasar, belum diukur):** sekitar 29 detik per contoh berarti 1.200 contoh ≈ 9–10 jam per epoch di T4. Ini melebihi sesi Colab gratis dan mendukung rencana memakai GPU IT Del (ADR-0012).

---

## 8. Tabel angka penting

### 8.1 Token visual per frame (diukur dengan processor)
| Sisi terpanjang | Ukuran (tanpa potong) | Token/frame | Dengan pita dipotong (F002–F027) |
|---|---|---|---|
| asli 1804 px | 1804 × 1080 | 1.904 | — |
| 1024 px | 1024 × 613 | 608 | 1024 × 429 → **416** |
| 768 px | 768 × 460 | 336 | 768 × 322 → **240** |
| 512 px | 512 × 307 | 160 | — |

F001 tidak berpita, sehingga tetap 608 (1024 px) atau 336 (768 px).

### 8.2 Total token contoh QG per konfigurasi (diukur, `--dry`)
| Konfigurasi | Total token | Keterangan |
|---|---|---|
| 1024 px × 27 frame, tanpa potong | 18.658 | Tidak mungkin di T4 |
| 768 px × 14, tanpa potong | 6.738 | OOM (terbukti) |
| 768 px × 14, potong | 5.490 | Di atas anggaran |
| 768 px × 10, potong | 4.466 | Di atas anggaran |
| **768 px × 9, potong** | **4.210** | Cadangan |
| 512 px × 14, tanpa potong | 4.274 | Muat (terbukti), tetapi buram |
| 1024 px × 6, potong | 4.594 | Di atas anggaran |
| **1024 px × 5, potong** | **4.162** | **Pilihan pilot, muat (terbukti)** |

Teks (system + metadata + label frame + transkrip + instruksi + jawaban) sekitar 1.900–2.000 token (4.162 − 2.272 = 1.890 untuk 5 frame; 4.274 − 2.240 = 2.034 untuk 14 frame, karena tiap frame menambah label teks); sisanya token gambar.

---

## 9. Keputusan yang tercatat (ADR)

| No | Keputusan | Status |
|---|---|---|
| 0001 | Checkpoint `Qwen/Qwen3.5-9B` (post-trained) | Diterima |
| 0002 | Keluaran hanya QG dan AE, dua tahap | Diterima |
| 0003 | QLoRA 4-bit untuk pilot | Sementara |
| 0004 | Vision encoder dibekukan, LoRA di bagian bahasa | Sementara |
| 0005 | Target modul LoRA: opsi (B), semua proyeksi linear bahasa di 32 lapisan | Diusulkan (dipakai di pilot) |
| 0006 | Masukan: frame terurut + transkrip ASR + instruksi | Diterima |
| 0007 | Script sendiri dengan Transformers + PEFT | Diterima |
| 0008 | `enable_thinking=False` | Diterima |
| 0009 | Keluaran JSON, kunci huruf, metadata waktu dari kode | Diusulkan |
| 0010 | Soal dan pembahasan berbahasa Indonesia | Diterima |
| 0011 | Split per kelompok sumber | Diterima |
| 0012 | Pembagian lingkungan: laptop, Colab, GPU IT Del | Sementara |
| 0013 | Prapemrosesan pilot: FFmpeg dan faster-whisper | Sementara |
| 0014 | Aturan kerja agent AI | Diterima |
| **0015** | Kontrak prompt dan JSON QG/AE v0.1 | Sementara (2 Nov) |
| **0016** | Potong pita hitam, frame merata, 1024 px × 5 frame | Sementara (3 Nov) |

---

## 10. Masalah, penyebab, dan penyelesaian

| Masalah | Penyebab | Penyelesaian | Pelajaran |
|---|---|---|---|
| Transkrip salah dengar ("Knya") | Keterbatasan ASR untuk istilah | Koreksi manual | ASR harus diperiksa manusia sebelum dipakai |
| Satu contoh 18,6 ribu token | 27 frame resolusi penuh | Batasi resolusi dan jumlah frame | Token gambar mendominasi memori |
| Kunci jawaban selalu terpanjang | Draf AI | Opsi diseimbangkan (v2) | Tinjauan pihak kedua menemukan pola tersembunyi |
| `HFValidationError` saat memuat model | Ejaan folder Drive (`Levelearn` vs `LeveLearn`) | Diagnostik `os.listdir` | Path harus dicek, tidak ditebak |
| `cd: /content/repo: No such file` | Sel dijalankan sebelum unzip | Ulangi urutan sel | Urutan sel di Colab penting |
| GPU tidak tersedia | Jatah GPU gratis habis | Menunggu | Colab gratis tidak menjamin GPU |
| bf16 dipakai di T4 | `is_bf16_supported()` melapor emulasi | Pilih dtype dari compute capability | Fitur "didukung" belum tentu native |
| OOM 768 px × 14 frame | Memori aktivasi pada 6.738 token (penyebab rinci belum diukur) | Kurangi token; logits hemat | Ukur dulu, jangan menebak |
| `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | Dugaan: pip meng-upgrade torch lewat torchvision | Runtime bersih, pip tanpa torchvision; cadangan `--sin-cudnn` | Jangan meng-upgrade torch bawaan Colab |
| 512 px muat tetapi buram | Batas memori T4 | Potong pita + kurangi frame → 1024 px × 5 | Area kosong pun memakan token |
| Folder huruf kecil di disk | Folder diganti nama di Windows | Perbaikan nama setelah unzip di Colab | Linux membedakan huruf besar-kecil |

---

## 11. Berkas yang dihasilkan dan fungsinya

### Script (`scripts/`)
| Berkas | Fungsi |
|---|---|
| `transkrip.py` | ASR segmen pilot → `transkrip.txt/json` |
| `prompt_qg_ae.py` | Prompt, pemilihan frame, penyusun pesan Layout A, validator, konversi LeveLearn |
| `cek_sampel.py` | Muat dan potong gambar, bangun contoh, ukur token visual, periksa masking dan template |
| `bangun_contoh_latih.py` | Contoh acuan → `segmen.json` + `contoh_latih.jsonl` |
| `uji_1_batch.py` | Uji QLoRA 1 batch di GPU: LoRA, maju-mundur, ablasi, sweep, catatan JSON |

### Data (`Data/Pilot/`, diabaikan git)
`segmen_0410_0627.mp4`, `frames/frame_001–027.jpg`, `transkrip.txt` (terkoreksi), `transkrip.json` (mentah), `segmen.json`, `contoh_acuan_pilot.json` (v2, diperiksa), `contoh_latih.jsonl`.

### Dokumen
- `docs/adr/0001–0016` dan `docs/adr/README.md`
- `docs/STATUS.md` (serah terima antarsesi)
- `docs/research/Riset_Prompt_QG_AE_dan_Masukan_VLM_LeveLearn.md`
- `docs/pertanyaan/Pertanyaan_Format_QG_AE.md`
- `docs/tinjauan/response_json.md`
- `.claude/skills/qwen35-vlm-finetune/references/format-data.md` (§2–4 diperbarui sesuai ADR-0015)

### Catatan uji
`uji_1_batch_2026-11-03_014908.json` (percobaan 3) dan `uji_1_batch_2026-11-03_022152.json` (percobaan 4), saat ini di root repo; tempat yang disiapkan adalah `runs/`. Salinan juga ada di Drive (`MyDrive/LeveLearn/runs_uji/`).

### Perubahan yang belum di-commit (per 3 Nov)
`scripts/uji_1_batch.py` (opsi `--sin-cudnn`, `--sweep max_side:jumlah_frame`, potong pita, pemeriksaan bukti), `scripts/cek_sampel.py` (potong pita), `scripts/prompt_qg_ae.py` (`pilih_frame`), `docs/adr/0016-…md`, `docs/adr/README.md`, `docs/STATUS.md`, dokumen ini, dan kedua berkas JSON uji.

---

## 12. Batasan: yang belum terbukti

- **Training sungguhan belum dijalankan.** Memori dengan optimizer, penurunan loss, dan stabilitas fp16 selama banyak langkah belum diketahui.
- **Adapter belum pernah disimpan, dimuat ulang, atau dipakai untuk generasi.** Belum ada keluaran soal dari model 9B.
- **Belum ada perbandingan model dasar dan adapter.** Itu butuh data uji terpisah dari sumber lain.
- **Data hanya 1 segmen dari 1 video** (4 contoh). Cukup untuk menguji alur, tidak cukup untuk melatih model yang baik atau menarik kesimpulan.
- **Keterbacaan frame dinilai dari mata manusia**, bukan dari kemampuan model membaca notasi.
- **Penyebab cuDNN mismatch masih dugaan** (kuat, tetapi belum dibuktikan langsung).
- **Pengaruh `flash-linear-attention` belum diukur.**
- Angka di lingkungan campuran (percobaan 1–2) **tidak dipakai** sebagai hasil final.

---

## 13. Langkah berikutnya

1. **Mini-training pilot**: beberapa langkah optimizer pada 4 contoh, untuk membuktikan memori cukup dengan optimizer dan loss latih turun. Ini bukan bukti kualitas, karena model dilatih dan diuji pada contoh yang sama.
2. **Simpan adapter → muat ulang di proses baru → generasi** dengan prompt dan pengaturan sama, lalu periksa keluaran mentah dengan validator. Keluaran mentah disimpan terpisah dari hasil parsing.
3. (Opsional) **Uji satu faktor `flash-linear-attention`** dengan konfigurasi sama persis, untuk melihat pengaruh pada memori dan waktu.
4. **Data tambahan** dari video lain dengan pembagian split per sumber, lalu evaluasi model dasar dan adapter pada data uji terpisah.
5. **Penyelarasan dokumen**: proposal V1 perlu disesuaikan dengan ADR-0002, 0003, 0015, dan 0016; skema LeveLearn untuk menyimpan AE belum diputuskan.
