# Progres 3 November 2026 — Mini-Training Pilot, Simpan dan Muat Ulang Adapter Qwen3.5-9B (QG + AE) LeveLearn

Dokumen ini melanjutkan `docs/progress/2_november_2026.md`. Dokumen sebelumnya berhenti pada **uji 1 batch** (percobaan 4): konfigurasi pilot terbukti muat di GPU, tetapi bobot model belum pernah diubah. Dokumen ini mencatat apa yang dikerjakan sesudahnya: script mini-training disusun berdasarkan instruksi berbasis rujukan, lalu dijalankan di Google Colab sampai adapter tersimpan, dimuat ulang, dan dipakai untuk membuat soal. Semua angka dan masalah yang muncul dicatat beserta artinya.

Aturan baca (sama dengan dokumen sebelumnya):
- **Terverifikasi** = sudah dijalankan dan keluarannya dilihat.
- **Perkiraan/dugaan** = belum diukur atau belum dibuktikan; ditandai eksplisit.
- Angka GPU berasal dari berkas yang ditulis script di Colab dan dikirim pengguna (zip folder run dari Drive, kini disalin ke `runs/pilot_02/`). Satu-satunya angka `pilot_01` yang tersedia adalah keluaran layar yang ditempel pengguna.

---

## Daftar isi

1. [Ringkasan singkat](#1-ringkasan-singkat)
2. [Perbandingan dengan kondisi akhir 2 November](#2-perbandingan-dengan-kondisi-akhir-2-november)
3. [Glosarium istilah baru](#3-glosarium-istilah-baru)
4. [Lingkungan](#4-lingkungan)
5. [Gambaran alur teknis (diperbarui)](#5-gambaran-alur-teknis-diperbarui)
6. [Kronologi pekerjaan rinci](#6-kronologi-pekerjaan-rinci)
7. [Hasil rinci run `pilot_02`](#7-hasil-rinci-run-pilot_02)
8. [Analisis: apa arti angka-angka itu](#8-analisis-apa-arti-angka-angka-itu)
9. [Status milestone M1–M3](#9-status-milestone-m1m3)
10. [Masalah, penyebab, dan penyelesaian](#10-masalah-penyebab-dan-penyelesaian)
11. [Berkas yang dihasilkan dan fungsinya](#11-berkas-yang-dihasilkan-dan-fungsinya)
12. [Batasan: yang belum terbukti](#12-batasan-yang-belum-terbukti)
13. [Langkah berikutnya](#13-langkah-berikutnya)

---

## 1. Ringkasan singkat

Hari ini, untuk **pertama kalinya, seluruh alur fine-tuning berjalan dari awal sampai akhir** pada segmen pilot (1 segmen, 4 contoh latih), di Colab Tesla T4:

1. Disusun `Instruksi_Claude_Mini_Training_Qwen35_LeveLearn.md` (spesifikasi berbasis rujukan R01–R17 dan dokumentasi resmi T01–T05), lalu script `scripts/latih_pilot.py` ditulis mengikutinya. Ada juga 14 uji unit (`scripts/uji_unit_latih_pilot.py`) yang lolos di laptop.
2. Colab dijalankan dari **akun Google kedua** karena jatah GPU akun pertama habis. Model disalin ke `MyDrive/Model/Qwen3.5-9B`.
3. **Run `pilot_01` OOM** (kehabisan memori GPU) pada percobaan ke-14 setelah 5 update. Adapter tidak tersimpan.
4. **Run `pilot_02` berhasil** setelah satu setelan memori ditambahkan (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), tanpa mengubah konfigurasi lain:
   - 8 update optimizer dari 16 percobaan. Percobaan 1–8 dilewati oleh mekanisme pengaman fp16.
   - Loss latih berbobot turun **0,754 → 0,274**.
   - 400 tensor adapter berubah, semuanya finite (tidak ada NaN/inf).
   - Adapter tersimpan (160 MB).
5. **Muat ulang di proses baru lolos**: bobot identik (selisih 0,0), loss identik (selisih 0,0), dan soal yang dihasilkan identik.
6. **Soal sesudah latih bermasalah.** Bentuk JSON tiap soal sudah sesuai kontrak (berbeda dari model dasar), tetapi model **tidak berhenti di 3 soal**. Model menulis q1, q2, …, q6 dengan pertanyaan yang hampir sama sampai terpotong di batas 1.024 token, sehingga JSON tidak bisa dibaca.

Kesimpulan jujur: **mekanik alur sudah terbukti**, yaitu update, simpan, muat ulang, dan generasi. **Kualitas belum bisa disimpulkan.** Model dilatih dan diuji pada contoh yang sama, datanya hanya 1 segmen, dan keluarannya sendiri belum benar.

---

## 2. Perbandingan dengan kondisi akhir 2 November

Bagian ini mencocokkan isi `2_november_2026.md` dengan kondisi sekarang, supaya jelas mana yang sudah selesai, mana yang berubah, dan mana yang masih terbuka.

### 2.1 Butir "belum dilakukan" di dokumen 2 November
| Butir di dokumen 2 Nov (§1, §12, §13) | Kondisi 3 Nov | Bukti |
|---|---|---|
| Training sungguhan (langkah optimizer) belum dijalankan | ✅ **Selesai**: 8 update AdamW | `runs/pilot_02/train_steps.jsonl`, `adapter_update_check.json` |
| Memori dengan optimizer belum diketahui (perkiraan 0,3–0,5 GiB) | ✅ **Terukur**: optimizer menambah **0,299 GiB**, cocok dengan hitungan teori (§8.3) | `train_steps.jsonl` (puncak allocated) |
| Adapter belum pernah disimpan atau dimuat ulang | ✅ **Selesai**: disimpan, lalu dimuat ulang di proses baru dengan hasil identik | `reload_check.json` |
| Belum ada keluaran soal dari model 9B | ✅ **Ada**: QG model dasar dan QG sesudah latih | `baseline_qg.raw.txt`, `after_qg.raw.txt` |
| Stabilitas fp16 selama banyak langkah belum diketahui | ✅ **Terlihat**: 8 langkah awal dilewati (normal untuk fp16), sesudahnya stabil; semua bobot finite | §8.1 |
| Belum ada perbandingan model dasar vs adapter | ⚠️ **Sebagian**: ada perbandingan pada contoh latih yang sama. Ini **bukan evaluasi** karena butuh data uji dari sumber lain | §7.6 |
| Generasi AE (pembahasan) | ⬜ **Belum dijalankan** (opsi `--gen-ae` tidak dipakai) | `summary.json`: `ae_generation_pending` |
| Uji `flash-linear-attention` | ⬜ Belum; kernel masih fallback | `environment.json` |
| Data hanya 1 segmen | ⬜ Masih sama | — |
| Penyebab cuDNN mismatch masih dugaan | ⬜ Masih dugaan; runtime bersih tetap tanpa error (cuDNN aktif di `pilot_02`) | `environment.json` |

### 2.2 Hal yang sudah berubah dari isi dokumen 2 November
- **§4.2 Lokasi model**: dulu `MyDrive/LeveLearn/models/Qwen3.5-9B` di akun pertama. Sekarang di akun kedua: `MyDrive/Model/Qwen3.5-9B`.
- **§11 Catatan uji**: dulu dua berkas JSON uji 1 batch ada di root repo. Sekarang keduanya sudah di `runs/` bersama `uji_1_batch_2026-11-03_percobaan2_output.md` (dulu `output.md`).
- **§11 Perubahan yang belum di-commit**: semua yang tercantum di sana sudah di-commit pengguna (`0a9390f`, `454863e`, `be393d4`, `0197b65`).
- **§7 Perkiraan waktu**: sekitar 30 detik per langkah latih kini **terukur** dalam training sungguhan (28,7–31,4 detik), bukan hanya pada uji 1 batch.
- **§8.1 Ukuran F001**: preflight mencatat F001 diperkecil menjadi 1024 × **614**, sedangkan tabel lama menulis 1024 × 613. Selisih 1 piksel ini kemungkinan berasal dari pembulatan setelah pemotongan pita. Jumlah token tidak berubah (QG tetap 4.162).

---

## 3. Glosarium istilah baru

Istilah dari dokumen 2 November (loss, LoRA, QLoRA, fp16, OOM, dan lainnya) tidak diulang. Di bawah ini istilah yang baru muncul hari ini, diurutkan menurut kemunculannya dalam alur.

| Istilah | Arti sederhana | Kaitan dengan proyek |
|---|---|---|
| **Mini-training** | Training yang sangat singkat (beberapa langkah saja) untuk membuktikan alur bekerja, bukan untuk menghasilkan model yang bagus | 8 update pada 4 contoh |
| **Preflight** | Pemeriksaan sebelum "terbang": data dan prompt dicek tanpa memuat model | Tahap A script |
| **Baseline** | Hasil model dasar (belum dilatih) sebagai pembanding | `baseline_qg.*` |
| **Generasi** | Menyuruh model menulis jawabannya sendiri, token demi token, **tanpa** diberi jawaban acuan | QG sebelum dan sesudah latih |
| **Greedy decoding** | Pada setiap langkah generasi, model selalu memilih token dengan peluang tertinggi. Hasilnya pasti (bisa diulang persis) | `do_sample=False`, `num_beams=1` |
| **`max_new_tokens`** | Batas jumlah token yang boleh ditulis model dalam satu generasi | 1.024 |
| **EOS (end of sequence)** | Token penanda "jawaban selesai" | `<|im_end|>` = ID 248046 |
| **Teacher forcing** | Cara menghitung loss: model diberi jawaban acuan dan dinilai seberapa yakin ia menebak tiap token berikutnya. Berbeda dengan generasi, di mana model menulis sendiri | Semua angka loss |
| **Optimizer AdamW** | Algoritma yang mengubah bobot berdasarkan gradien. Ia menyimpan dua "catatan" per parameter (rata-rata gradien dan rata-rata kuadratnya) | `lr=1e-4`, `betas=(0,9; 0,999)`, `weight_decay=0` |
| **Learning rate (lr)** | Besar langkah perubahan bobot per update | 0,0001, konstan tanpa *warmup* |
| **Optimizer state** | Catatan AdamW yang disimpan di memori GPU (2 angka fp32 per parameter yang dilatih) | Terukur 0,299 GiB |
| **Update / langkah dilewati** | *Update* = bobot benar-benar diubah. *Dilewati* = gradien rusak (inf/NaN) sehingga bobot sengaja tidak diubah | 8 update, 8 dilewati |
| **Percobaan** | Satu kali maju-mundur pada satu contoh, berakhir sebagai update atau dilewati | 16 percobaan |
| **AMP (Automatic Mixed Precision)** | Sebagian hitungan dikerjakan dalam 16-bit supaya hemat memori dan cepat, sisanya tetap 32-bit | `torch.autocast` fp16 |
| **GradScaler / loss scaling** | fp16 tidak bisa menyimpan angka yang sangat kecil (bisa jadi 0). Karena itu loss dikalikan angka besar (*scale*) sebelum backward, lalu gradien dibagi kembali. Kalau *scale* terlalu besar, gradien meluap menjadi inf; langkah itu dilewati dan *scale* dibagi dua | 65.536 → 256 |
| **Overflow / inf / NaN** | *Overflow*: angka melebihi batas format. *inf*: tak hingga. *NaN*: "bukan angka" (misalnya hasil inf − inf) | Norm gradien NaN di percobaan 1–8 |
| **Norm gradien** | Satu angka yang merangkum seberapa besar semua gradien | 0,89–2,01 pada update |
| **Gradient clipping** | Bila norm gradien > batas, semua gradien diperkecil sebanding supaya update tidak terlalu besar | Batas 1,0 |
| **Epoch** | Satu putaran melewati seluruh data latih | 8 update = 2 epoch atas 4 contoh |
| **VRAM allocated vs reserved** | *Allocated*: memori yang benar-benar dipakai tensor. *Reserved*: memori yang sudah "dipesan" PyTorch dari GPU (termasuk ruang kosong yang belum dipakai) | Reserved mencapai 14,29 dari 14,56 GiB |
| **Fragmentasi memori** | Total memori kosong cukup, tetapi terpecah menjadi potongan kecil sehingga satu blok besar tidak muat | Dugaan penyebab OOM `pilot_01` |
| **`expandable_segments`** | Setelan pengelola memori PyTorch: memori dipesan sebagai wilayah yang bisa membesar, bukan potongan-potongan terpisah, sehingga fragmentasi berkurang. Hitungan model tidak berubah | Kunci keberhasilan `pilot_02` |
| **State dict** | Daftar semua tensor bobot beserta namanya | Dibandingkan sebelum/sesudah dan setelah muat ulang |
| **safetensors** | Format berkas bobot yang aman dan cepat dibaca | `adapter_model.safetensors` (160 MB) |
| **Reload / muat ulang** | Menyalakan proses Python baru, memuat model dasar, lalu memasang adapter dari berkas. Tujuannya membuktikan bahwa yang tersimpan benar-benar hasil latih | Tahap G |
| **Determinisme** | Konfigurasi dan seed yang sama memberi angka yang sama persis | Terlihat pada `pilot_01` vs `pilot_02` (§8.5) |
| **Seed** | Angka awal pengacak; seed yang sama membuat inisialisasi acak (misalnya matriks LoRA A) sama | 42 |
| **Hash SHA-256** | "Sidik jari" berkas; berkas berubah sedikit saja, hash-nya berubah total | `data_manifest.json` mencatat hash data dan script |
| **Parsing ketat** | Membaca teks keluaran sebagai JSON **tanpa memperbaiki** apa pun. Kalau rusak, dicatat rusak | `parse_ketat` |
| **Skema / validator** | Pemeriksaan bentuk JSON terhadap kontrak (ADR-0015): jumlah soal, opsi A–D, kunci, `evidence_ids` | `validasi_qg` |
| **Degenerasi / pengulangan** | Generasi yang terjebak mengulang pola yang sama terus-menerus | Soal q3–q6 sesudah latih |
| **Milestone M1–M3** | Tiga tahap penerimaan di instruksi: M1 update dan simpan, M2 generasi dan validasi, M3 persistensi (muat ulang) | §9 |

---

## 4. Lingkungan

### 4.1 Laptop (tidak berubah)
Windows 11, venv `.venv`: torch 2.15.0+cpu, transformers 5.22.1, peft 0.21.2. Dipakai untuk menulis script, menjalankan uji unit, dan preflight `--dry`.

### 4.2 Google Colab: akun kedua (terverifikasi dari `environment.json` `pilot_02`)
| Komponen | Nilai |
|---|---|
| Akun | Akun Google kedua (jatah GPU gratis akun pertama habis) |
| GPU | Tesla T4, **14,56 GiB** total, compute capability 7.5 |
| Python | 3.13.16 |
| Paket | torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 — **sama** dengan lingkungan percobaan 4 |
| CUDA / cuDNN | 13.0 / `92700` (9.27), cuDNN aktif tanpa `--sin-cudnn` |
| Kernel cepat | `flash-linear-attention` dan `causal-conv1d` **tidak** terpasang (fallback PyTorch) |
| Setelan memori | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (hanya `pilot_02`) |
| Lokasi model | `MyDrive/Model/Qwen3.5-9B` |
| Lokasi run | `MyDrive/runs_pilot/pilot_01` dan `pilot_02` |
| Sidik jari config model | SHA-256 `d0883072…32b05` (dicatat supaya reload memakai model dasar yang sama) |

Catatan:
- Runtime pertama di akun kedua ternyata **runtime CPU** (torch `2.11.0+cpu`, `nvidia-smi` tidak ada) karena tipe runtime belum diganti ke T4. Setelah diganti ke T4, lingkungannya sama dengan percobaan 4.
- Versi Python 3.13 belum tercatat pada percobaan 1–4, jadi belum diketahui apakah Python di sana sama.

---

## 5. Gambaran alur teknis (diperbarui)

```
(Bagian data sama dengan dokumen 2 Nov: segmen → 27 frame + transkrip → manifest → contoh acuan diperiksa → contoh_latih.jsonl)
                  │
                  ▼
 latih_pilot.py --mode train
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ A  PREFLIGHT (tanpa model)   status anotasi, pemetaan AE↔QG, label = target, │ ✅
 │                              template sama, target tidak bocor ke prompt     │
 │ B  MUAT MODEL 4-bit + GENERASI BASELINE QG (tanpa jawaban acuan)            │ ✅
 │ C  PASANG LoRA (r=16, 200 modul, 32 lapisan) + snapshot bobot awal          │ ✅
 │ D  UJI KESETARAAN loss hemat = loss standar  +  LOSS AWAL 4 contoh          │ ✅
 │ E  MINI-TRAINING: AdamW + GradScaler fp16, sampai 8 update berhasil         │ ✅ (pilot_02)
 │ F  LOSS AKHIR → SIMPAN adapter → GENERASI QG sesudah latih                  │ ✅ (keluaran terpotong)
 └──────────────────────────────────────────────────────────────────────────┘
                  │  adapter/adapter_model.safetensors
                  ▼
 latih_pilot.py --mode reload  (PROSES BARU)
 G  muat model dasar yang sama + adapter dari berkas → bandingkan bobot, loss, generasi   ✅ identik
                  │
                  ▼
 ⬜ generasi AE   ⬜ data video lain + split per sumber   ⬜ evaluasi model dasar vs adapter pada data uji
```

---

## 6. Kronologi pekerjaan rinci

### 6.1 Instruksi mini-training (`Instruksi_Claude_Mini_Training_Qwen35_LeveLearn.md`)
Pengguna menyusun instruksi rinci berbasis rujukan yang bisa dipertanggungjawabkan, dan script ditulis mengikuti instruksi itu. Isi pokoknya:

- **Rujukan inti (R01–R13)**, antara lain:
  - Hu dkk. 2021 (LoRA): matriks B diinisialisasi nol, bobot utama dibekukan.
  - Dettmers dkk. 2023 (QLoRA): 4-bit NF4 dan double quantization.
  - Choi dkk. 2026 (VLM kimia dengan Qwen3.5-9B).
  - Stamatakis dkk. 2025 (QG dari video edukasi, perbandingan sebelum/sesudah fine-tuning).
  - QuizTools, QGEval, RQUGE, dan Moore dkk. (penilaian soal).
- **Alternatif yang dibaca tetapi sengaja tidak dipakai (R14–R17)**: DoRA, LoftQ, PiSSA, AdaLoRA. Alasannya, uji alur pertama tidak boleh menambah variabel metode.
- **Dokumentasi teknis (T01–T05)**: model card Qwen3.5-9B, dokumentasi Transformers Qwen3.5, dokumentasi PEFT (kuantisasi dan format checkpoint), dan contoh AMP PyTorch 2.11 (autocast, GradScaler, unscale, clipping, langkah dilewati).
- **Peringatan penting (§3.3)**: angka di paper (misalnya lr 5e-5 di R03, 1e-6 di R05) berasal dari konteks lain. Karena itu lr 1e-4, r=16, 8 update, dan 5 frame pada pilot harus disebut **"pilihan teknik yang diuji"**, bukan "pengaturan terbaik menurut jurnal".
- **Rancangan tahap A–G** dan **milestone penerimaan M1–M3** (§9 dokumen ini).
- **Larangan klaim**: tidak boleh menyebut "model sudah paham", "berhasil membuat soal" sebelum generasi dilihat, atau "0,6722 adalah kemiripan".

### 6.2 Script `scripts/latih_pilot.py`
Satu script dengan dua mode. Fungsi dari `uji_1_batch.py`, `cek_sampel.py`, dan `prompt_qg_ae.py` dipakai ulang, sehingga cara membangun contoh **persis sama** dengan uji 1 batch yang sudah terbukti.

**Mode `train`:**

| Tahap | Yang dikerjakan | Mengapa |
|---|---|---|
| **A. Preflight** | Untuk 4 contoh: memeriksa status anotasi `diperiksa`; AE merujuk soal QG yang benar (`question_id` dan kunci sama); label yang dinilai = teks target; template latih = inferensi; **target tidak muncul di prompt**; `evidence_ids` ada di frame/transkrip yang ditampilkan; jendela waktu transkrip. Mencatat ukuran frame dan jumlah token. Dengan `--dry`, script berhenti di sini tanpa memuat model | Menangkap kesalahan data sebelum memakai GPU yang mahal |
| Catatan run | `config_resolved.json` (semua setelan, termasuk optimizer, LoRA, dan SHA-256 config model), `environment.json` (versi paket, GPU, cuDNN, setelan memori), `data_manifest.json` (hash data dan script) | Reproduksibilitas: run bisa dilacak dan diulang |
| **B. Baseline** | Memuat model 4-bit, lalu **generasi QG oleh model dasar** sebelum LoRA dipasang. Greedy, tanpa jawaban acuan, berhenti di EOS atau 1.024 token. Keluaran disimpan sebagai `.raw.txt` (teks mentah), `.tokens.json`, `.parsed.json`, dan `.validation.json` | Pembanding "sebelum"; keluaran mentah dipisah dari hasil parsing (aturan AGENTS.md) |
| **C. LoRA** | Memasang LoRA (regex ADR-0005 opsi B). Memeriksa bahwa **hanya** parameter LoRA yang bisa dilatih, semuanya fp32, dan matriks B awal nol. Mencatat daftar parameter (`trainable_parameters.json`) dan menyimpan *snapshot* bobot awal | Memastikan yang dilatih memang adapter, bukan model dasar |
| **D. Kesetaraan dan loss awal** | (1) Pada contoh pendek (AE q1, 2 frame, 512 px), loss "hemat" (logits hanya di posisi jawaban) dibandingkan dengan loss standar; toleransi 0,001. (2) Loss awal 4 contoh dalam mode evaluasi | Membuktikan trik hemat memori tidak mengubah angka loss |
| **E. Training** | Contoh diputar berurutan (QG, AE1, AE2, AE3, QG, …). Satu percobaan: maju dengan autocast fp16 → `scaler.scale(loss).backward()` → `unscale_` → clipping norm 1,0 → `scaler.step` → `scaler.update`. Langkah dianggap **update** hanya jika norm finite dan *scale* tidak turun; selain itu **dilewati**. Berhenti setelah 8 update, atau gagal setelah 32 percobaan. Setiap percobaan dicatat ke `train_steps.jsonl` | Membedakan percobaan, update, dan langkah dilewati sesuai instruksi §5 E |
| Cek perubahan | Membandingkan bobot adapter akhir dengan snapshot: berapa tensor berubah, perubahan maksimum, semua finite? → `adapter_update_check.json` | Bukti nyata bahwa bobot berubah, bukan hanya "optimizer dipanggil" |
| **F. Akhir** | Loss akhir 4 contoh (cara sama dengan loss awal) → `loss_before_after.json`. Menyimpan adapter (`safetensors`) + processor, membaca berkas kembali dan membandingkan. Lalu **generasi QG sesudah latih** dengan setelan sama persis dengan baseline. Adapter disimpan **sebelum** generasi supaya hasil latih tidak hilang bila generasi OOM | Sesuai instruksi §5 F butir 7 |
| Ringkasan | `summary.json` (status tiap tahap + waktu dan memori per fase), `review_template.md` (lembar pemeriksaan manual) | Status otomatis tidak membuktikan soal benar |

Script menolak folder output yang sudah berisi `adapter/` atau `train_steps.jsonl`, supaya run lama tidak tertimpa.

**Mode `reload`** (dijalankan di proses Python baru):
1. Membaca `config_resolved.json` run, lalu memastikan SHA-256 config model dan dtype sama.
2. Memuat model dasar dan memasang adapter dari berkas (`PeftModel.from_pretrained`, `is_trainable=False`).
3. Membandingkan bobot adapter yang dimuat dengan isi berkas, dan memastikan B tidak nol.
4. Menghitung loss 4 contoh, lalu membandingkannya dengan loss akhir saat training.
5. Generasi QG, lalu membandingkannya dengan `after_qg.raw.txt` (kalau berbeda, posisi token pertama yang berbeda dicatat).
6. Menulis `reload_check.json` dan memperbarui `summary.json`.

### 6.3 Uji unit (`scripts/uji_unit_latih_pilot.py`): 14/14 lolos di laptop
| Kelompok | Yang diuji |
|---|---|
| Parsing ketat | JSON valid terbaca; JSON rusak → `tidak_terbaca` + pesan galat; JSON berpagar Markdown dicatat dan **tidak diperbaiki** |
| Pemetaan AE | Data pilot benar; AE dengan kunci berbeda dari QG ditolak; AE dengan `question_id` salah ditolak |
| Bukti | Konfigurasi yang tidak menampilkan F027 (padahal dirujuk) ditolak |
| Deteksi langkah dilewati | Norm inf → dilewati; *scale* turun → dilewati; normal → update; tanpa scaler (bf16) → update |
| GradScaler sungguhan (CPU) | Gradien inf membuat step dilewati, bobot tidak berubah, dan terdeteksi sebagai dilewati |
| Loss hemat | Pada model palsu kecil, loss hemat = loss standar (selisih < 10⁻⁶) |
| Loss berbobot | Rata-rata berbobot jumlah token dihitung benar (1,75) |

Setelah satu baris ditambahkan (setelan memori dicatat di `environment.json`), uji diulang dan tetap 14/14.

### 6.4 Preflight lokal (`--dry`)
| Contoh | Total token | Token gambar | Token prompt | Token dinilai |
|---|---|---|---|---|
| QG | 4.162 | 2.272 | 3.455 | 707 |
| AE q1 | 4.054 | 2.272 | 3.636 | 418 |
| AE q2 | 4.099 | 2.272 | 3.662 | 437 |
| AE q3 | 4.126 | 2.272 | 3.683 | 443 |

Semua lolos: label = target, template sama, target tidak di prompt. Prompt AE lebih panjang daripada prompt QG karena AE juga membawa JSON soalnya; sebaliknya target AE lebih pendek.

**Satu peringatan data**: potongan transkrip T015 tercatat 135,0–138,0 detik relatif, melewati durasi segmen (137 detik). Angka itu berasal dari keluaran mentah Whisper dan tidak diubah. T015 juga tidak dirujuk sebagai bukti.

### 6.5 Persiapan Colab di akun kedua
1. Akun pertama terkena batas pemakaian GPU, jadi pengguna memakai akun Google lain dan menyalin model ke `MyDrive/Model/Qwen3.5-9B`.
2. Sel Colab disusun ulang dari nol. Path model dan run cukup ditulis sekali di variabel `MODEL` dan `RUN`, lalu dipakai sel lain lewat `{MODEL}` / `{RUN}`. Ada juga sel pemeriksa kelengkapan model: semua shard di `model.safetensors.index.json` ada, `config.json` ada, dan ukuran berkas terlihat.
3. **Masalah**: sel cek cuDNN gagal dengan `AssertionError: Torch not compiled with CUDA enabled`. Diagnosis: `nvidia-smi` tidak ada, dan semua paket torch berakhiran `+cpu`. Artinya runtime-nya **CPU**, bukan rusak karena pip. Solusi: *Runtime → Change runtime type → T4 GPU*, lalu ulangi dari awal.
4. Urutan sel: `nvidia-smi` → mount Drive → cek model → unzip + betulkan nama folder (`data/pilot` → `Data/Pilot`) → catat versi → `pip install` **tanpa torchvision** → cek conv3d cuDNN → uji unit → preflight `--dry` → train → reload.

### 6.6 Run `pilot_01`: OOM pada percobaan 14
Keluaran layar terakhir yang dilaporkan pengguna:
```
[13] pilot_bdd_01_s001_qg: loss 0.4389 | update | norm 0.893 | 31.4 dtk | puncak 13.57 GiB
training: {'status': 'OOM pada percobaan 14', 'update_berhasil': 5, 'percobaan': 14, 'target_tercapai': False, 'semua_finite': True} | tensor berubah: 400
Training tidak selesai: OOM pada percobaan 14. Lihat train_steps.jsonl
```
Arti:
- Training sudah berjalan dan bobot berubah (400 tensor, semuanya finite), tetapi berhenti di update ke-5 dari 8.
- Percobaan 14 adalah contoh **AE q1**, yang lebih pendek daripada QG di percobaan 13. Tetap saja percobaan itu OOM.
- Sesuai rancangan, script **tidak** menandai training sukses, **tidak** menyimpan adapter, dan tahap F tidak dijalankan.
- Dugaan penyebab: **fragmentasi memori**. Puncak allocated 13,57 GiB, sedangkan pada `pilot_02` puncak *reserved* mencapai 14,29 GiB dari 14,56 GiB. Artinya ruang tersisa sangat sedikit, dan bila terpecah-pecah, satu permintaan blok besar bisa gagal walaupun totalnya cukup. Dugaan ini **tidak dibuktikan langsung**, karena `train_steps.jsonl` `pilot_01` tidak diperiksa. Bukti tidak langsungnya: run dengan setelan anti-fragmentasi berhasil tanpa perubahan lain.
- Folder `pilot_01` tetap disimpan di Drive sebagai catatan run gagal dan tidak ditimpa.

**Keputusan perbaikan**: hanya **satu faktor** yang diubah, yaitu setelan `expandable_segments:True`. Setelan ini mengatur cara PyTorch memesan memori tanpa mengubah hitungan model. Resolusi, jumlah frame, LoRA, learning rate, dan optimizer sengaja tetap sama, supaya kalau berhasil penyebabnya jelas. Script ditambah satu baris agar setelan ini tercatat di `environment.json`. Alternatif yang **ditunda** karena masing-masing mengubah konfigurasi penelitian: *scale* awal GradScaler lebih kecil, optimizer 8-bit, atau jumlah frame lebih sedikit.

### 6.7 Run `pilot_02`: selesai
Dijalankan ke folder baru (`runs_pilot/pilot_02`) dengan perintah:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/latih_pilot.py --mode train --model "{MODEL}" --output-dir "{RUN}"
```
Keluaran akhir di layar:
```
training: {'status': 'selesai', 'update_berhasil': 8, 'percobaan': 16, 'target_tercapai': True, 'semua_finite': True} | tensor berubah: 400
loss akhir (berbobot): 0.27407109449034617
after QG: {'generation_status': 'berhasil', 'alasan_berhenti': 'batas_max_new_tokens', 'jumlah_token_baru': 1024, 'parse_status': 'tidak_terbaca', 'schema_status': 'tidak_diperiksa'}
Selesai.
```
Mode train berjalan dari 07:24:02 sampai 07:39:08 waktu Colab (kemungkinan UTC), yaitu **15 menit 6 detik**. Rincian hasil ada di §7.

### 6.8 Muat ulang (`--mode reload`): lolos
```
{"status": "selesai", "state": {"kunci_sama": true, "selisih_maks": 0.0, "lora_B_nonnol": true},
 "selisih_loss": 0.0, "generasi": "berhasil", "sama_dengan_after": true}
```
Muncul empat peringatan transformers: `causal_conv1d` dan `flash-linear-attention` tidak terpasang, sehingga dipakai implementasi PyTorch yang "benar tetapi jauh lebih lambat". Ini sudah diketahui sejak percobaan 1 dan tidak memengaruhi kebenaran hasil.

### 6.9 Pengambilan hasil dan kerapian repo
1. Pengguna mengunduh folder `pilot_02` dari Drive sebagai zip. Zip itu diekstrak ke `runs/pilot_02/` lalu dihapus.
2. `.gitignore` ditambah tiga aturan:
   - `.agents/`: salinan `.claude/skills`, dengan isi duplikat.
   - `*.zip`: semua arsip (bungkusan Colab dan unduhan run).
   - `runs/**/processor/`: salinan tokenizer 20 MB yang bisa dibuat ulang dari model.

   Bobot adapter tetap tidak dilacak karena aturan lama `runs/**/adapter*/` dan `*.safetensors`.
3. Pengguna membuat 4 commit:
   - `9dad750` chore(gitignore)
   - `6955852` feat(scripts): latih_pilot + uji unit + instruksi
   - `afa7e71` chore(runs): catatan pilot_02
   - `1bd937f` docs(status)
4. `docs/STATUS.md` diperbarui dengan hasil `pilot_01`, `pilot_02`, reload, dan analisis berkas.

---

## 7. Hasil rinci run `pilot_02`

Semua angka di bagian ini **terverifikasi** dari berkas `runs/pilot_02/`.

### 7.1 Konfigurasi (`config_resolved.json`)
| Kelompok | Nilai |
|---|---|
| Data | 4 contoh (1 QG + 3 AE), `source_group_id` `pilot_bdd_source_01` |
| Gambar | 1024 px, 5 frame (F001, F007, F014, F021, F027), pita hitam dipotong (ADR-0016) |
| Model | 4-bit NF4 + double quant, `sdpa`, fp16, gradient checkpointing (`use_reentrant=False`) |
| LoRA | r=16, alpha=32, dropout 0,05, regex ADR-0005 opsi B, LoRA standar (B awal nol) |
| Optimizer | AdamW, lr 1e-4 konstan tanpa *warmup*, betas (0,9; 0,999), eps 1e-8, weight decay 0 |
| Batch | 1 contoh, tanpa *gradient accumulation* |
| Clipping | norm maksimum 1,0 |
| Target | 8 update berhasil, maksimum 32 percobaan |
| Generasi | greedy, `max_new_tokens` 1.024, EOS `<|im_end|>`, `enable_thinking=False`, hanya QG (`gen_ae=false`) |
| Seed | 42 |

### 7.2 Parameter yang dilatih (`trainable_parameters.json`)
- **400 tensor** (200 modul × matriks A dan B), **40.108.032 parameter**, semua fp32, di lapisan 0–31. Angka ini sama dengan uji 1 batch.
- Berkas ini menulis `jumlah_parameter_total` = 5.765.080.304, berbeda dari 9.449.921.776 pada percobaan 1. **Dugaan penjelasan** (belum diverifikasi): bobot 4-bit bitsandbytes disimpan dua angka per byte, sehingga penghitungan elemen tensor menghitung bobot terkuantisasi hanya separuhnya. Persentase parameter yang dilatih tetap dilaporkan memakai angka 9,45 miliar (0,4244%).

### 7.3 Waktu dan memori per fase (`summary.json`)
| Fase | Waktu | Puncak allocated | Puncak reserved |
|---|---|---|---|
| Muat model dari Drive | 87 dtk | 7,37 GiB (7,34 setelah muat) | 7,40 GiB |
| Generasi baseline QG (872 token) | 90 dtk | 9,64 GiB | 9,75 GiB |
| Loss awal 4 contoh (evaluasi) | 31 dtk | 10,70 GiB | 10,85 GiB |
| Training: per percobaan | 28,7–31,4 dtk | 13,00–13,56 GiB | 14,13–14,29 GiB |
| Generasi QG sesudah latih (1.024 token) | 158 dtk | 9,79 GiB | 9,90 GiB |
| **Total mode train** | **15 mnt 6 dtk** | | |

Kecepatan generasi: baseline sekitar **9,7 token/detik**, sesudah latih sekitar **6,5 token/detik**. **Dugaan**: perlambatan berasal dari lapisan LoRA yang dihitung terpisah di setiap modul (adapter tidak digabung ke bobot dasar). Ini belum diukur secara terpisah.

### 7.4 Langkah training (`train_steps.jsonl`)
| Percobaan | Contoh | Loss | *Scale* sebelum → sesudah | Norm gradien | Status | Update kumulatif |
|---|---|---|---|---|---|---|
| 1 | QG | 0,6722 | 65.536 → 32.768 | NaN | dilewati | 0 |
| 2 | AE q1 | 0,8446 | 32.768 → 16.384 | NaN | dilewati | 0 |
| 3 | AE q2 | 0,7976 | 16.384 → 8.192 | NaN | dilewati | 0 |
| 4 | AE q3 | 0,7542 | 8.192 → 4.096 | NaN | dilewati | 0 |
| 5 | QG | 0,6722 | 4.096 → 2.048 | NaN | dilewati | 0 |
| 6 | AE q1 | 0,8446 | 2.048 → 1.024 | NaN | dilewati | 0 |
| 7 | AE q2 | 0,7976 | 1.024 → 512 | NaN | dilewati | 0 |
| 8 | AE q3 | 0,7542 | 512 → 256 | NaN | dilewati | 0 |
| 9 | QG | 0,6722 | 256 → 256 | 1,563 | **update** | 1 |
| 10 | AE q1 | 0,8054 | 256 → 256 | 2,006 | **update** | 2 |
| 11 | AE q2 | 0,6983 | 256 → 256 | 1,371 | **update** | 3 |
| 12 | AE q3 | 0,6214 | 256 → 256 | 1,427 | **update** | 4 |
| 13 | QG | 0,4389 | 256 → 256 | 0,893 | **update** | 5 |
| 14 | AE q1 | 0,4476 | 256 → 256 | 1,263 | **update** | 6 |
| 15 | AE q2 | 0,4125 | 256 → 256 | 1,211 | **update** | 7 |
| 16 | AE q3 | 0,3754 | 256 → 256 | 1,128 | **update** | 8 |

Hasil yang tercatat: tidak ada gradien kosong, tidak ada loss non-finite, dan semua update terjadi di percobaan 9–16. Puncak reserved tertinggi 14,29 GiB.

### 7.5 Loss sebelum dan sesudah (`loss_before_after.json`)
Diukur dengan cara yang sama di awal dan akhir: mode evaluasi, teacher forcing, 4 contoh latih.

| Contoh | Token dinilai | Loss awal | Loss akhir |
|---|---|---|---|
| QG | 707 | 0,6722 | 0,2750 |
| AE q1 | 418 | 0,8446 | 0,2610 |
| AE q2 | 437 | 0,7976 | 0,2665 |
| AE q3 | 443 | 0,7542 | 0,2924 |
| **Berbobot semua** | 2.005 | **0,7536** | **0,2741** |
| Berbobot QG / AE | | 0,6722 / 0,7979 | 0,2750 / 0,2735 |

Rumus berbobot: `Σ(loss_i × token_i) / Σ token_i`. Contoh dengan jawaban lebih panjang berpengaruh lebih besar.

**Uji kesetaraan (tahap D)**: pada AE q1 versi pendek (2 frame, 512 px, 2.006 token), loss hemat = loss standar = 0,83536, dengan selisih **0,0**. Trik `logits_to_keep` terbukti tidak mengubah loss pada model sungguhan, bukan hanya pada model palsu di uji unit.

### 7.6 Perubahan bobot adapter (`adapter_update_check.json`)
| Kelompok | Tensor berubah | Perubahan maksimum (absolut) | Norm perubahan |
|---|---|---|---|
| Semua | 400 / 400 | 0,000806 | 2,341 |
| LoRA A | 200 / 200 | 0,000640 | 1,498 |
| LoRA B | 200 / 200 | 0,000806 | 1,799 |

Semua finite. Berkas menyimpan catatan rasio kunjungan: "1 QG : 3 AE per putaran (objective tidak otomatis seimbang)". Artinya dalam setiap putaran, model melihat tugas AE tiga kali lebih sering daripada QG.

### 7.7 Generasi QG: model dasar vs sesudah latih
Kedua generasi memakai prompt yang sama (5 frame + transkrip + instruksi QG), tanpa jawaban acuan, greedy, dengan batas 1.024 token.

| Aspek | Baseline (model dasar) | Sesudah latih (adapter) |
|---|---|---|
| Berhenti karena | EOS (selesai sendiri), 872 token | **Batas 1.024 token** (terpotong) |
| Bentuk teks | JSON berindentasi (banyak baris) | JSON satu baris, seperti contoh latih |
| Parsing | Terbaca | **Tidak terbaca**: `Expecting property name enclosed in double quotes` di kolom 3047 (teks terputus di tengah objek) |
| `status` | `insufficient_context` (konteks tidak cukup), **tetapi tetap memberi 3 soal**, sehingga kontradiktif | `ok` |
| ID soal | `Q001`, `Q002`, `Q003` (melanggar kontrak) | `q1`, `q2`, … (sesuai kontrak) |
| Opsi | Daftar `["A. …", "B. …"]`, padahal kontrak meminta objek (melanggar kontrak) | Objek `{"A": …, "B": …}` (sesuai kontrak) |
| Jumlah soal | 3 | **Tidak berhenti**: q1–q5 utuh, q6 terpotong |
| Kunci jawaban | A, A, A | A, A, A, A, A |
| Stimulus | 2 jenis (Q001 dan Q003 sama) | **Satu kalimat yang sama** di semua soal |
| Pertanyaan (stem) | Berbeda-beda | 5 stem, hanya **2 yang berbeda**; 4 di antaranya identik ("…yang menghasilkan nilai T…") |
| `evidence_ids` | 4–5 frame, tanpa ID transkrip | `F001`, `F007`, + satu potongan transkrip berbeda tiap soal |
| Validator | `tidak_sesuai`: "opsi harus tepat A–D dan terisi" (3×) | Tidak diperiksa karena JSON tidak terbaca |

**Pemeriksaan logika sederhana oleh agent** (belum diperiksa manusia; dicatat sebagai temuan awal):
- Soal q1 sesudah latih ("baris yang mewakili tepat empat kombinasi bernilai T") dengan kunci `p = T, q = *, r = *` **benar**: bila p = T, maka p ∨ (q ∧ r) selalu T untuk keempat kombinasi q dan r.
- Soal q2 ("…bernilai F") dengan kunci `p = F, q = *, r = *` **salah**: bila p = F, nilainya sama dengan q ∧ r, yang bernilai T saat q = T dan r = T. Jadi baris itu tidak mewakili empat kombinasi bernilai F.
- Isi soal sesudah latih **bukan salinan** contoh acuan. Di acuan, q1 menanyakan arti tanda `*` (kunci A), q2 baris ringkas untuk baris 1–2 (kunci D), dan q3 syarat sah penggabungan (kunci B).

### 7.8 Muat ulang (`reload_check.json`)
| Pemeriksaan | Hasil |
|---|---|
| Proses baru | Ya |
| Nama tensor berkas = nama tensor yang dimuat | Ya |
| Selisih nilai bobot maksimum | 0,0 |
| LoRA B bukan nol (adapter berisi hasil latih) | Ya |
| Loss berbobot reload vs loss akhir di memori | 0,27407 vs 0,27407 → selisih **0,0** (per contoh juga identik) |
| Generasi QG reload vs sesudah latih | **Identik** (`indeks_beda_pertama: null`); 1.024 token, 156 dtk |
| Lingkungan | Sama dengan training (torch 2.11.0+cu130, fp16, expandable_segments) |

---

## 8. Analisis: apa arti angka-angka itu

### 8.1 Mengapa 8 langkah pertama dilewati
- fp16 hanya bisa menyimpan angka sampai sekitar 65.504. Supaya gradien yang sangat kecil tidak menjadi nol, GradScaler mengalikan loss dengan *scale* awal **65.536** (2¹⁶), yaitu nilai bawaan PyTorch.
- Dengan pengali sebesar itu, gradien meluap menjadi inf, sehingga norm-nya NaN. GradScaler lalu **sengaja melewati** langkah itu (bobot tidak diubah) dan membagi *scale* dengan dua.
- Setelah 8 kali dibagi dua (2¹⁶ → 2⁸ = **256**), gradien tidak lagi meluap dan semua langkah berikutnya menjadi update.
- Ini perilaku **normal dan sesuai rancangan** AMP, bukan kerusakan. Dampaknya: sekitar 8 × 30 detik = **4 menit terbuang**, dan total percobaan menjadi 16.
- Bukti bahwa langkah 1–8 benar-benar tidak mengubah apa pun: loss di percobaan 5–8 **sama persis** dengan percobaan 1–4 (0,6722; 0,8446; …), dan juga sama dengan loss awal mode evaluasi. Kesamaan dengan mode evaluasi terjadi karena B = 0, sehingga dropout di jalur LoRA belum berpengaruh.
- `pilot_01` juga mencapai 5 update tepat di percobaan 13, sama dengan `pilot_02`. Ini konsisten: pola yang sama, 8 langkah dilewati di awal.

### 8.2 Loss turun: apa artinya, apa yang bukan artinya
- Pada kunjungan kedua (percobaan 13–16), loss latih tiap contoh sudah lebih rendah daripada kunjungan pertama. Contohnya QG 0,672 → 0,439 dan AE q3 0,754 → 0,375.
- Loss akhir mode evaluasi (0,274) lebih rendah daripada loss di percobaan 13–16 karena dua hal: diukur **sesudah** seluruh 8 update (loss percobaan dihitung sebelum update-nya sendiri), dan dropout mati di mode evaluasi.
- **Arti yang sah**: alur belajar bekerja. Gradien mengalir ke adapter, optimizer mengubah bobot ke arah yang menurunkan loss, dan hasilnya terbaca kembali.
- **Bukan bukti kualitas.** Model dinilai pada **4 contoh yang sama** dengan yang dipelajarinya (2 kali). Turunnya loss di sini sebagian besar adalah tanda model mulai menghafal pola contoh. Dalam teacher forcing, loss 0,274 berarti rata-rata peluang token benar sekitar e^(−0,274) ≈ 0,76. Ini **bukan** angka kemiripan atau ketepatan soal.

### 8.3 Memori: tambahan optimizer terukur dan cocok dengan teori
Bandingkan puncak allocated pada contoh yang sama sebelum dan sesudah optimizer AdamW mulai bekerja. *State* AdamW baru dibuat pada update pertama di percobaan 9; langkah yang dilewati tidak membuatnya.

| Contoh | Sebelum ada state AdamW | Sesudah ada state AdamW | Selisih |
|---|---|---|---|
| QG | 13,2574 GiB (percobaan 1) | 13,5562 GiB (percobaan 13) | **0,2988 GiB** |
| AE q1 | 13,0046 GiB (percobaan 2) | 13,3034 GiB (percobaan 14) | **0,2988 GiB** |

Hitungan teori: AdamW menyimpan 2 angka fp32 (4 byte) per parameter yang dilatih, yaitu 40.108.032 × 2 × 4 byte = 320.864.256 byte = **0,2988 GiB**. Hasil ukur **cocok sampai empat angka desimal**. Jadi tambahan memori training sungguhan terhadap uji 1 batch sudah dapat dijelaskan sepenuhnya. Perkiraan di dokumen 2 November (0,3–0,5 GiB) berada di batas bawahnya.

Konsekuensinya, sisa ruang di T4 sangat tipis. Puncak reserved 14,29 dari 14,56 GiB berarti sisa sekitar 0,27 GiB. Konfigurasi yang sedikit lebih besar (lebih banyak frame, jawaban lebih panjang, atau r lebih besar) kemungkinan besar tidak muat di T4.

### 8.4 Besar perubahan bobot cocok dengan cara kerja AdamW
- Pada AdamW, satu update menggeser tiap parameter **kira-kira sebesar learning rate** (sekitar 0,0001), apa pun besar gradiennya. Dengan 8 update, perubahan maksimum yang diharapkan sekitar 8 × 0,0001 = 0,0008. Hasil ukur LoRA B: **0,000806**.
- LoRA A berubah lebih sedikit (maksimum 0,000640, di bawah 7 × 0,0001 = 0,0007). **Penjelasan yang cocok dengan angka** (tidak diuji terpisah): pada update pertama B masih nol, sehingga gradien A tepat nol dan A belum bergeser. A baru bergerak mulai update kedua, jadi paling banyak 7 kali.
- Kesimpulannya, angka perubahan bobot konsisten dengan teori dan tidak ada yang "meledak".

### 8.5 Determinisme: `pilot_01` dan `pilot_02` memberi angka yang sama
Pada percobaan 13 (QG), `pilot_01` mencatat loss 0,4389 dan norm 0,893. `pilot_02` mencatat loss 0,43886 dan norm 0,8927, **sama**. Puncak allocated hampir sama (13,57 vs 13,56 GiB). Artinya:
- Dengan seed dan konfigurasi yang sama, run ini berulang persis (paling tidak sampai percobaan 13).
- Setelan `expandable_segments` **tidak mengubah hitungan**, hanya cara memori dipesan. Ini mendukung keputusan untuk menganggap `pilot_02` sebagai run yang setara dengan `pilot_01` minus OOM.

Generasi reload yang identik dengan generasi sesudah latih juga menunjukkan determinisme pada lingkungan yang sama. Instruksi mencatat bahwa greedy tidak menjamin hasil identik **lintas backend**, misalnya GPU atau versi yang berbeda.

### 8.6 Mengapa soal sesudah latih tidak berhenti di 3
Yang **terlihat** (fakta):
- Model sudah menangkap **bentuk** kontrak: opsi objek A–D, id `q1`, `status: ok`, JSON satu baris seperti contoh latih, dan `evidence_ids` dari ID yang ditampilkan.
- Model **tidak** menangkap aturan **jumlah** (tepat 3 soal lalu `]}` dan EOS). Ia terus menambah soal dengan stimulus sama, stem hampir sama, dan kunci selalu A. Ini pola pengulangan (degenerasi).
- Isi soal tidak menyalin acuan, dan satu kunci salah secara logika (q2).

Yang **belum terbukti** (dugaan, perlu diuji bila dianggap penting):
- Contoh QG hanya **satu** dan hanya dilihat **dua kali** dalam update, sedangkan AE dilihat 6 kali. Sinyal "berhenti setelah soal ketiga" sangat lemah.
- Learning rate konstan 1e-4 pada data sangat sedikit dapat menggeser perilaku generasi lebih cepat daripada yang bisa dipulihkan oleh 2 kunjungan.
- Greedy decoding diketahui rentan mengulang pola begitu satu pola mulai muncul.

Yang **tidak** akan membantu: menaikkan `max_new_tokens`. Pengulangan hanya akan bertambah panjang. Begitu juga menyetel ulang hiperparameter **pada contoh latih ini lalu menilai di contoh yang sama**, karena itu sama dengan "mengajar untuk ujian".

### 8.7 Model dasar pun belum mengikuti kontrak
Baseline berhenti sendiri dan menulis JSON yang terbaca, tetapi melanggar kontrak di tiga hal: opsi berbentuk daftar, id `Q001`, dan `status: insufficient_context` yang bertentangan dengan adanya 3 soal. Ini mendukung alasan fine-tuning: model dasar tidak otomatis patuh pada format LeveLearn, dan adapter memang mengubah bentuk keluarannya. Pada catatan `issue`-nya, model dasar juga "mengeluh" bahwa segmen hanya mendukung satu konsep (reduksi tabel kebenaran). Ini masukan yang layak diperhatikan untuk aturan "3 soal berbeda konsep" pada segmen pendek.

---

## 9. Status milestone M1–M3

Milestone diambil dari instruksi §11.

| Milestone | Syarat | Status |
|---|---|---|
| **M1 — Update dan simpan** | 4 contoh lolos preflight; hanya adapter yang dilatih; ada update nyata dan jumlahnya diketahui; bobot berubah dan finite; loss awal/akhir sebanding; adapter tersimpan dan terbaca | ✅ **Tercapai** (`pilot_02`) |
| **M2 — Generasi dan validasi** | Baseline dan sesudah latih dijalankan tanpa target dengan setelan sebanding; mentah, token, parsing, dan validator terpisah; tidak ada klaim "soal benar" dari JSON valid; generasi AE dicoba atau ditandai pending | ⚠️ **Sebagian**: QG lengkap dan terdokumentasi (termasuk kegagalan parsing sesudah latih); AE ditandai `ae_generation_pending` |
| **M3 — Persistensi** | Proses baru memuat adapter tersimpan; state, loss, dan keluaran diperiksa dengan model dasar yang sama | ✅ **Tercapai** |

---

## 10. Masalah, penyebab, dan penyelesaian

| Masalah | Penyebab | Penyelesaian | Pelajaran |
|---|---|---|---|
| Jatah GPU akun pertama habis | Batas pemakaian Colab gratis | Pindah ke akun Google kedua; model disalin ke `MyDrive/Model/Qwen3.5-9B` | Siapkan lokasi model di lebih dari satu akun bila memakai Colab gratis |
| `Torch not compiled with CUDA enabled` | Runtime masih CPU (torch `+cpu`, tidak ada `nvidia-smi`), bukan rusak karena pip | Ganti tipe runtime ke T4, ulangi dari awal | Selalu jalankan `nvidia-smi` di sel pertama |
| `pilot_01` OOM di percobaan 14 (5/8 update) | Dugaan kuat: fragmentasi memori; sisa ruang ±0,3 GiB setelah state AdamW 0,3 GiB | `expandable_segments:True`, satu faktor saja, folder run baru | Uji 1 batch belum mewakili training: optimizer menambah memori dan fragmentasi bertambah seiring langkah |
| 8 langkah awal dilewati | *Scale* awal GradScaler 65.536 terlalu besar untuk gradien model ini di fp16 | Dibiarkan (perilaku normal), tercatat rinci | Bedakan percobaan, update, dan langkah dilewati; jangan menghitung "langkah" dari percobaan |
| Soal sesudah latih tidak berhenti di 3, JSON terpotong | Belum terbukti; dugaan: 1 contoh QG × 2 kunjungan terlalu lemah, ditambah pengulangan greedy | Belum diperbaiki; dicatat sebagai temuan pilot | Bentuk format lebih cepat dipelajari daripada aturan jumlah; kualitas harus dinilai pada data uji terpisah |
| Zip run berisi adapter 160 MB dan tokenizer 20 MB | Folder run diunduh utuh | `.gitignore`: `*.zip`, `runs/**/processor/` (adapter sudah diabaikan) | Simpan catatan kecil di git, bobot di Drive |

---

## 11. Berkas yang dihasilkan dan fungsinya

### Script dan instruksi
| Berkas | Fungsi |
|---|---|
| `Instruksi_Claude_Mini_Training_Qwen35_LeveLearn.md` | Spesifikasi mini-training berbasis rujukan (R01–R17, T01–T05), tahap A–G, milestone M1–M3 |
| `scripts/latih_pilot.py` | Mode `train` (A–F) dan `reload` (G); semua artefak run |
| `scripts/uji_unit_latih_pilot.py` | 14 uji tanpa GPU |

### Catatan run (`runs/pilot_02/`, dilacak git)
| Berkas | Isi |
|---|---|
| `config_resolved.json` | Semua setelan run, termasuk optimizer, LoRA, frame, dan SHA-256 config model |
| `environment.json` | Python, versi paket, GPU, CUDA/cuDNN, setelan memori |
| `data_manifest.json` | Urutan contoh, hash SHA-256 data dan script, jumlah token |
| `preflight.json` | Hasil tahap A per contoh, ukuran frame, peringatan T015 |
| `trainable_parameters.json` | 400 nama tensor LoRA, jumlah parameter, dtype, lapisan |
| `train_steps.jsonl` | Satu baris per percobaan: loss, *scale*, norm, status, waktu, memori |
| `adapter_update_check.json` | Perubahan bobot adapter (jumlah tensor berubah, maksimum, norm) |
| `loss_before_after.json` | Loss awal dan akhir per contoh dan berbobot |
| `baseline_qg.{raw.txt, tokens.json, parsed.json, validation.json}` | Generasi model dasar: mentah, token, parsing, validasi |
| `after_qg.{…}` | Generasi sesudah latih (empat berkas yang sama) |
| `reload_qg.{…}` | Generasi setelah muat ulang (identik dengan `after_qg`) |
| `reload_check.json` | Hasil tahap G |
| `summary.json` | Ringkasan semua tahap, waktu dan memori per fase |
| `review_template.md` | Lembar pemeriksaan manual: sesuai sumber, dapat dijawab, 1 kunci benar, pengecoh masuk akal, notasi tepat |

**Tidak dilacak git** (ada di Drive dan salinan lokal): `adapter/adapter_model.safetensors` (160 MB), `adapter/adapter_config.json`, `processor/`. Folder `pilot_01` hanya ada di Drive.

### Dokumen
`docs/STATUS.md` (diperbarui) dan dokumen ini.

---

## 12. Batasan: yang belum terbukti

- **Kualitas soal belum dinilai.** Satu-satunya soal sesudah latih rusak (terpotong, berulang, satu kunci salah). Belum ada pemeriksaan manusia; `review_template.md` belum diisi.
- **Perbandingan model dasar vs adapter belum sah sebagai evaluasi.** Perbandingan dilakukan pada contoh latih itu sendiri. Evaluasi butuh segmen dari video lain yang dipisah per sumber (ADR-0011).
- **Generasi AE belum pernah dijalankan.** Training memakai contoh AE, tetapi kemampuan model menulis pembahasan belum diuji.
- **Penyebab OOM `pilot_01` dan penyebab pengulangan soal masih dugaan.**
- **Data tetap 1 segmen dari 1 video** (4 contoh). Cukup untuk membuktikan alur, tidak cukup untuk melatih model yang baik.
- **Kernel `flash-linear-attention` / `causal-conv1d` belum dipasang**, jadi pengaruhnya pada waktu dan memori belum diukur.
- **Pengulangan persis baru terbukti di lingkungan yang sama** (T4, versi paket yang sama), belum lintas GPU.
- **Hiperparameter** (lr 1e-4, r 16, 8 update, 5 frame) adalah pilihan teknik yang diuji, bukan hasil optimasi dan bukan "terbaik menurut jurnal".

---

## 13. Langkah berikutnya

Urutan saran; keputusan tetap di tangan pengguna:

1. **Tahap data (saran utama).** Siapkan segmen dari video lain beranotasi yang diperiksa manusia, lalu bagi train/val/test per sumber video. Tanpa ini, kualitas tidak bisa dinilai dan perbaikan masalah "tidak berhenti di 3 soal" tidak bisa diuji dengan jujur.
2. **(Opsional) Run `pilot_03`** dengan perubahan yang dicatat sebagai run baru:
   - Jalankan `--gen-ae` supaya generasi AE baseline dan sesudah latih juga tercatat (melengkapi M2).
   - Pertimbangkan *scale* awal GradScaler yang lebih kecil (misalnya 256), yang menghemat sekitar 4 menit per run. Ini perubahan konfigurasi, jadi perlu disetujui dulu dan dicatat terpisah.
3. **Isi `review_template.md`** untuk `baseline_qg` dan `after_qg` sebagai latihan prosedur pemeriksaan manual. Hasilnya juga bisa dipakai sebagai contoh di dokumen TA.
4. **(Opsional) Uji satu faktor `flash-linear-attention`.** Dengan sisa memori hanya ±0,27 GiB di T4, kernel ini bisa menentukan apakah konfigurasi yang lebih besar mungkin dijalankan.
5. **GPU IT Del.** Training 1 segmen saja sudah memakan sekitar 15 menit dan memori T4 hampir penuh. Data yang lebih banyak membutuhkan GPU yang lebih besar (ADR-0012).
6. **Penyelarasan dokumen** (sama dengan sebelumnya): proposal V1 perlu disesuaikan dengan ADR-0002, 0003, 0015, dan 0016; skema LeveLearn untuk AE belum diputuskan. Bila langkah 2 mengubah konfigurasi pilot, keputusan itu dicatat sebagai ADR baru.
