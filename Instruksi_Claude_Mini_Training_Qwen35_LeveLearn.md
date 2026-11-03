# Instruksi untuk Claude: Implementasi dan Verifikasi Mini-Training Qwen3.5-9B untuk QG–AE LeveLearn

Tanggal penyusunan: 3 November 2026  
Tujuan: membuat kode pilot yang dapat diperiksa, ditelusuri ke sumber, dan diuji di Colab; bukan menetapkan konfigurasi eksperimen final.

> Berikan seluruh berkas ini kepada Claude yang mengerjakan repositori LeveLearn. Bagian analisis sumber menjadi dasar keputusan; bagian spesifikasi merupakan instruksi implementasi. Dokumen ini tidak menyatakan bahwa mini-training sudah dijalankan.

## 1. Mandat dan batas pekerjaan

Implementasikan `scripts/latih_pilot.py` untuk membuktikan alur:
**generasi model dasar → mini-training QLoRA → bukti perubahan adapter → penyimpanan → generasi sesudah latihan → parsing dan validasi → pemuatan ulang adapter pada proses baru.**

Gunakan kode proyek yang sudah ada. Jangan menulis ulang pipeline tanpa alasan, mengganti model, atau diam-diam mengubah kontrak QG/AE.

Yang ingin dibuktikan pada pilot:
1. Empat contoh multimodal dapat melewati pembaruan optimizer dengan numerik dan memori yang terukur.
2. Parameter adapter benar-benar berubah, sedangkan bobot dasar dibekukan.
3. Adapter dapat disimpan, dimuat ulang secara mandiri, dan dipakai menghasilkan keluaran.
4. Keluaran mentah dapat diperiksa tanpa menyembunyikan kegagalan format, pemotongan, atau isi.

Pilot satu segmen bukan bukti peningkatan kualitas, generalisasi, atau manfaat bagi mahasiswa. Perubahan jawaban juga bukan syarat mutlak bahwa parameter berhasil diperbarui; sebaliknya, turunnya loss tidak membuktikan soal benar.

## 2. Apa yang sudah tersedia dan sudah terbukti

### 2.1 Sumber proyek yang harus dibaca sebelum mengedit

Baca versi aktual:
- `AGENTS.md`, skill proyek `qwen35-vlm-finetune`, `docs/STATUS.md`, dan ADR yang relevan, khususnya 0001–0008, 0011–0016 jika tersedia.
- `scripts/prompt_qg_ae.py`, `scripts/cek_sampel.py`, `scripts/uji_1_batch.py`, `scripts/bangun_contoh_latih.py`.
- `Data/Pilot/segmen.json`, `Data/Pilot/transkrip.txt`, `Data/Pilot/contoh_latih.jsonl`.
- Catatan uji `uji_1_batch_2026-11-03_022152.json` dan catatan percobaan sebelumnya jika tersedia.
- Dokumen riset lokal dan contoh acuan yang sudah disetujui manusia.

Jangan mengklaim sudah membaca berkas yang tidak tersedia. Jika ZIP dipakai sebagai sumber, inventarisnya tidak memuat seluruh ADR atau skill repositori; jangan menganggap semuanya ada di ZIP.

### 2.2 Keadaan pilot saat instruksi ini disusun

| Komponen | Keadaan |
|---|---|
| Model | `Qwen/Qwen3.5-9B`, checkpoint post-trained; jangan tertukar dengan `-Base` |
| Segmen | Video asli detik 250–387, durasi 137 detik |
| Bahan | 27 frame dan 15 potongan transkrip terkoreksi |
| Contoh latih | 4 contoh: 1 QG menargetkan 3 soal; 3 AE masing-masing menargetkan satu pembahasan |
| Status anotasi | `diperiksa`; pemeriksa tercatat Moses |
| GPU uji | Tesla T4, sekitar 14,6 GiB |
| Konfigurasi yang terbukti | 1024 piksel sisi terpanjang, 5 frame, pita hitam dipotong |
| Frame pada uji QG | F001, F007, F014, F021, F027 |
| Panjang uji QG | 4.162 token total; 2.272 token gambar; 707 token target yang dinilai |
| Loss sebelum optimizer | 0,6722480058670044 |
| Ablasi | Frame asli 0,672248; gambar kosong 0,708612 |
| Memori puncak maju–mundur | 13,2763 GiB |
| Waktu maju–mundur | 29,3936 detik; bukan waktu generasi dan bukan waktu satu langkah training lengkap |
| LoRA saat uji | r=16, alpha=32, dropout=0,05; 32 lapisan bahasa; sekitar 40,1 juta parameter trainable |
| Yang belum terbukti | Memori optimizer, pembaruan parameter multimodal, generasi QG/AE, pemuatan ulang adapter |

707 token adalah serialisasi seluruh target QG, mencakup struktur JSON, tiga soal, opsi, kunci, bukti, dan penutup jawaban `<|im_end|>\n`. Bukan hanya tiga field `stem`, bukan token gambar, dan bukan pembahasan AE.

Loss 0,6722 adalah cross-entropy dengan teacher forcing: setiap posisi memprediksi token berikutnya menggunakan konteks serta token acuan sebelumnya. Itu bukan skor kemiripan hasil generasi, akurasi 67%, atau ukuran pemahaman.

Catatan lingkungan uji: torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0; verifikasi ulang runtime aktual. Riwayat konflik cuDNN terjadi setelah perubahan paket; jangan meng-upgrade torch/torchvision secara membabi buta.

### 2.3 Ketidakrapian paket yang harus ditangani

- ZIP menyimpan `data/pilot`, sedangkan kode dan manifest mencari `Data/Pilot`. Linux membedakan kapitalisasi. Sediakan pemeriksaan dan normalisasi yang eksplisit; jangan mengganti path acak.
- ZIP tidak membawa model 19 GB, video segmen, atau `contoh_acuan_pilot.json`. Uji/training memakai data olahan yang tersedia; tidak perlu menjalankan ulang ASR atau pembangun data.
- `contoh_latih_rapi.json` adalah salinan baca, bukan input kanonis. Gunakan JSONL aktual.
- Pada manifest/transkrip yang diperiksa, T015 berakhir pada detik relatif 138 sedangkan segmen dinyatakan 137 detik. Tandai ketidaksesuaian waktu dan cek sumber asli; jangan memperbaiki anotasi atau mengarang waktu secara diam-diam. Jangan klaim validitas temporal penuh selama belum diselesaikan. Jika aturan data proyek mewajibkan rentang ketat, hentikan preflight dengan pesan lokasi yang jelas.

## 3. Analisis riset di folder Drive: apa yang benar-benar mendukung keputusan

Folder sumber: https://drive.google.com/drive/folders/1x9cJ4lmjkyj7S5QnhDORYU2KlYcU1J2_

Inventaris folder pada tanggal pemeriksaan menampilkan **165 berkas**, termasuk kemungkinan duplikat, nama generik, dan satu proposal DOCX. Angka tersebut bukan 165 studi unik atau 165 artikel jurnal. Sebanyak **17 PDF relevan diakses sebagai teks**, lalu bagian metode/hasil yang relevan ditelaah; berkas lain hanya diinventarisasi berdasarkan metadata. Tidak dilakukan tinjauan sistematis atas seluruh isi folder, deduplikasi studi menyeluruh, atau verifikasi indeks jurnal.

### 3.1 Sumber inti dan batas penerapannya

| ID | Sumber dan lokasi bukti yang ditelaah | Dasar yang dapat dipakai | Batas untuk LeveLearn |
|---|---|---|---|
| R01 | Hu dkk., **LoRA: Low-Rank Adaptation of Large Language Models**, PDF v2 2021, §2 dan §4.1, Gambar 1 | Adaptasi melalui matriks tambahan dengan bobot utama dibekukan; objective prediksi autoregresif; inisialisasi B nol | Tidak menentukan regex Qwen3.5 atau rank terbaik untuk logika |
| R02 | Dettmers dkk., **QLoRA: Efficient Finetuning of Quantized LLMs**, PDF v1 2023, §3 dan §4–5 | Basis 4-bit, NF4, double quantization, dan adapter; membahas optimizer berpaging serta memori | PDF tersimpan adalah preprint v1; hasil LLM lain tidak menjamin VLM 9B muat di T4 |
| R03 | Choi, Lee, Kang, **Efficient fine-tuning of vision-language adapters in chemical VLMs for molecular image-text tasks**, Journal of Cheminformatics 18:109, 2026; bagian fine-tuning, Tabel 3–5 | Membandingkan strategi layer pada Qwen3.5-9B; konfigurasi L-MLP+visual merger unggul dalam tugas kimia yang diuji | Mendukung perlunya menguji pemilihan layer, bukan bukti visual merger wajib dilatih pada logika. Pilot tetap memakai target bahasa yang sudah disepakati |
| R04 | Stamatakis dkk., **Enhancing the Learning Experience: Using Vision-Language Models to Generate Questions for Educational Videos**, arXiv:2505.01790v1, 2025; §3.3, §4.2–4.3 | Perbandingan sebelum/sesudah fine-tuning, variasi modalitas, dan kebutuhan analisis manual pertanyaan | Model, data, dan jenis soal berbeda. Bukan dasar lima frame atau delapan langkah |
| R05 | Dutulescu dkk., **QuizTools: Question Generation with Explanations**, Algorithms 19(10):858, 2026; §3.2–3.3 dan §4–5 | Keluaran soal, kunci, pengecoh, penjelasan; penyaringan dan penyuntingan manusia | Paper memakai Llama 3.1 8B-Instruct dan prosedur berbeda. Tidak membuktikan skema JSON dua tahap kita optimal |
| R06 | Yu dkk., **Context Selection and Rewriting for Video-based Educational Question Generation**, AAAI-26; abstrak dan metode pemilihan/penulisan ulang konteks | Relevansi konteks transkrip dan frame serta keterkaitan dengan waktu dan jawaban | Metodenya dipandu target jawaban; QG kita tidak diberi kunci acuan sebagai input generasi |
| R07 | Berger dkk., **From Generation to Evaluation: A Resource for Error-Categorized Question Generation from Video Transcripts**, LREC 2026, hlm. 2166–2177; §4.1/Tabel 1 dan §5 | Analisis jenis kesalahan; pemisahan berdasarkan konteks menjaga pertanyaan satu konteks dalam split yang sama | Split per sumber video pada proyek adalah keputusan yang lebih ketat; kategori paper bukan pemeriksa notasi logika siap pakai |
| R08 | Fu dkk., **QGEval: Benchmarking Multi-dimensional Evaluation for Question Generation**, EMNLP 2024, hlm. 11783–11803; abstrak, §2, §3 | Penilaian fluency, clarity, conciseness, relevance, consistency, answerability, answer consistency | Bukan validator JSON atau metrik khusus operator logika |
| R09 | Mohammadshahi dkk., **RQUGE: Reference-Free Metric for Evaluating Question Generation by Answering the Question**, Findings ACL 2023, hlm. 6845–6867; abstrak, Gambar 1 dan metode | Kesamaan dengan satu acuan dapat gagal menilai pertanyaan valid; jawababilitas relevan | Tidak otomatis cocok untuk MCQ multimodal Bahasa Indonesia. Tidak perlu memuat model RQUGE pada pilot |
| R10 | Moore dkk., **Assessing the Quality of Multiple-Choice Questions Using GPT-4 and Rule-Based Methods**; abstrak dan metode item-writing flaws | Pemeriksaan aturan dapat melengkapi penilaian model/manusia | Validator struktur kita hanya menangani sebagian cacat; lolos validator tidak membuktikan kunci benar |
| R11 | Nagai dan Uto, **Automatic Distractor Generation in Multiple-Choice Questions Using Large Language Models with Expert-Informed Distractor Strategies**, ICCE 2025; abstrak dan metode | Pengecoh perlu masuk akal, berbeda, dan terkait kesalahan yang relevan | Tugas reading comprehension berbeda; jangan mengimpor kategori tanpa pemeriksaan pengajar logika |
| R12 | Fu dkk., **Video-MME**, PDF arXiv:2405.21075v3, 2025; abstrak dan rancangan benchmark | Menguji berbagai modalitas dan durasi; pentingnya anotasi | Benchmark video QA bukan pelatihan QG, dan tidak menetapkan jumlah frame kita |
| R13 | Liang dkk., **OmniEdu: Open Foundation Models for Learning and Teaching**, arXiv:2609.23088v1, 2026; §3, §4.1, konfigurasi tambahan | Kurasi data pendidikan dan pembandingan dengan base model; pemisahan kontaminasi evaluasi | Technical report/preprint; menggunakan antara lain model -Base dan full-parameter SFT. Bukan resep QLoRA post-trained di T4 |

### 3.2 Alternatif adaptasi yang dibaca tetapi tidak diaktifkan pada pilot

| ID | Paper | Temuan relevan dan keputusan |
|---|---|---|
| R14 | **DoRA: Weight-Decomposed Low-Rank Adaptation**, ICML 2024; abstrak dan pendekatan | Menawarkan pemisahan magnitude/direction. Simpan sebagai alternatif; jangan menambah variabel metode pada uji alur pertama |
| R15 | **LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models**, PDF v4 2023; abstrak dan metode | Inisialisasi bersama kuantisasi dapat berbeda dari LoRA standar. Jangan menggunakannya diam-diam karena mengubah asumsi adapter awal nol |
| R16 | **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models**; abstrak/metode awal | Inisialisasi berbasis komponen utama, bukan B nol standar. Tidak menjadi konfigurasi pilot ini |
| R17 | **AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning**, ICLR 2023; abstrak/metode | Rank/budget adaptif merupakan eksperimen terpisah, bukan kebutuhan membuktikan save/reload |

### 3.3 Tiga contoh agar tidak menyalahgunakan angka paper

- R03, Tabel 3: r=32, alpha=64, dropout=0,1, learning rate 5e-5, batch per device 4, accumulation 8, dua epoch pada tugas kimia.
- R05, §3.2: learning rate 1e-6, AdamW-8bit, batch 64 dengan accumulation, satu epoch pada Llama 3.1 8B-Instruct.
- R13, §4.1 dan konfigurasi tambahan: full-parameter SFT, learning rate 5e-6, tiga epoch.

Angka di atas menunjukkan **perbedaan konteks eksperimen**, bukan kandidat yang harus disalin. Learning rate 1e-4, r=16, delapan update, AdamW biasa, greedy decoding, dan lima frame pada pilot kita harus dijelaskan sebagai **pilihan teknik yang diuji**, bukan “pengaturan terbaik menurut jurnal”.

### 3.4 Keterlacakan sumber

Tautan berkas Drive untuk R01–R17 ada pada daftar sumber di bagian akhir. Identitas R03 diverifikasi pula melalui DOI/penerbit dan R05 melalui penerbit. Status publikasi sumber lain mengikuti identitas yang terbaca pada PDF; jangan menyebut seluruhnya jurnal atau mengklaim semuanya sudah peer-reviewed.

Jika hendak menulis klaim baru dalam komentar kode/proposal, baca bagian sumber yang mendukungnya. Jangan mengambil temuan dari judul berkas, mengarang nomor halaman, atau menganggap daftar pustaka ini sebagai bukti semua pilihan teknis.

## 4. Sumber teknis tambahan yang diperlukan

Gunakan dokumentasi resmi sebagai pendamping paper:
- T01: Model card [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- T02: [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5), terutama input multimodal, labels, cache, dan `logits_to_keep`.
- T03: [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), persiapan model kuantisasi dan target LoRA.
- T04: [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint), penyimpanan adapter dan dependensinya pada model dasar.
- T05: [PyTorch 2.11 AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), autocast, GradScaler, unscale, clipping, dan skipped steps.

Dokumentasi online dapat berbeda dari versi yang dipasang. Periksa signature dan kode paket runtime aktual sebelum memakai API. Jangan memaksakan contoh versi terbaru ke Colab yang sudah terbukti berjalan.

## 5. Rancangan pekerjaan yang harus diimplementasikan

### Tahap A — Preflight, tanpa menjalankan training

1. Verifikasi GPU, path, versi paket, kelas model, konfigurasi kuantisasi, serta ruang tulis output.
2. Baca empat baris JSONL; periksa task, status anotasi, pemetaan AE ke soal, dan keutuhan target.
3. Gunakan penyusun prompt dan pemilihan frame yang sama dengan pilot. Jangan membangun template percakapan sendiri.
4. Untuk setiap contoh, pastikan seluruh evidence target tersedia dalam frame terpilih atau transkrip yang benar-benar disajikan.
5. Hitung token untuk keempat contoh, bukan QG saja. Pastikan label hanya mencakup target, tidak prompt/gambar.
6. Verifikasi hasil decode token label sama dengan target asli plus penutup. Verifikasi kesesuaian template latih dan inferensi.
7. Catat frame IDs, ukuran sebelum/sesudah crop, token visual, token prompt, token target, serta total.
8. Jangan memotong target, rumus, opsi, atau transkrip secara otomatis untuk mengatasi panjang.
9. Buat mode `--dry` yang menjalankan pemeriksaan data/token tanpa memuat bobot 9B atau optimizer. Processor tetap diperlukan.
10. Catat sumber data dan hash berkas; jangan menyebut metadata “diperiksa” sebagai verifikasi ulang isi oleh kode.

**Keluaran:** laporan preflight dan konfigurasi aktual. Kesalahan fatal harus menunjuk berkas/example_id serta sebabnya.

### Tahap B — Muat model dan generasi pembanding

1. Muat model dasar 4-bit sekali selama proses training, dengan NF4 dan double quantization.
2. Pada T4 gunakan fp16 untuk komputasi; jangan memilih bf16 hanya karena fungsi deteksi melaporkan dukungan emulasi.
3. Lakukan generasi QG dengan **prompt dan gambar saja**, tanpa `target`, tanpa labels, tanpa opsi/kunci soal acuan tersisip sebagai contoh.
4. Pisahkan fungsi pembentukan batch training dan input generasi agar target tidak bocor.
5. Model `eval()`, gunakan `inference_mode()` atau padanannya. Atur cache/checkpointing sesuai fase dan kembalikan secara eksplisit.
6. Gunakan `do_sample=False`, `num_beams=1`, satu keluaran, dan `enable_thinking=False`. Jangan mengklaim identik bit demi bit di seluruh GPU/backend.
7. Anggaran awal QG `max_new_tokens=1024` boleh dipakai sebagai pilihan pilot yang dapat diubah. Target acuan sekitar 707 token tidak menjamin keluaran bebas selesai dalam 707 token.
8. Simpan raw text dan raw generated token IDs sebelum parsing. Potong input prompt dari output decode agar yang dinilai hanya token baru.
9. Catat jumlah token baru, EOS/stop reason jika dapat ditentukan, durasi, dan indikasi terkena batas token. Jangan mengklaim truncation pasti hanya dari panjang sama dengan limit bila sebab berhenti tidak diperiksa.
10. Setelah generasi, lepaskan output, cache, dan referensi tensor besar. Jangan mempertahankan scores seluruh vocab untuk seluruh keluaran bila tidak diperlukan.

**Keluaran:** `baseline_qg.raw.txt`, token IDs, dan laporan parsing/validasi. Bila generasi gagal/OOM, catat kegagalan dan fase; jangan menyebut baseline berhasil. Pilihan melanjutkan training tanpa baseline harus eksplisit dan diberi label hasil tidak lengkap.

### Tahap C — Pasang dan periksa adapter

1. Gunakan LoRA standar, bukan DoRA/PiSSA/LoftQ/AdaLoRA.
2. Pertahankan r=16, alpha=32, dropout=0,05 sebagai konfigurasi awal pilot agar perubahan dari uji sebelumnya minimal.
3. Reuse target regex `TARGET` pada `uji_1_batch.py` setelah memeriksa nama modul aktual.
4. Sasaran meliputi proyeksi bahasa yang sudah dipakai pada attention biasa, linear attention, dan MLP; jangan hanya q_proj/v_proj.
5. Periksa daftar seluruh trainable params, jumlah parameter, dtype, dan indeks layer yang tersentuh.
6. Pastikan hanya parameter adapter yang trainable. Vision encoder, visual merger, embedding, lm_head, dan mtp tetap tidak dilatih dalam konfigurasi pilot ini.
7. Tegaskan vision encoder tetap melakukan forward; “dibekukan” tidak berarti frame diabaikan.
8. Jadikan seluruh parameter trainable LoRA float32 jika dibutuhkan untuk AMP/optimizer yang dipilih; periksa actual dtype. Jangan mengubah seluruh model 9B menjadi float32.
9. Jangan memakai `prepare_model_for_kbit_training` tanpa memeriksa efeknya pada pembekuan, casting, input gradients, dan memori. Bila memakai jalur persiapan manual yang sama dengan pilot, dokumentasikan invariant yang dipenuhi dan perbedaannya dari helper resmi.
10. Pada LoRA standar awal, B bernilai nol. Jangan mewajibkan semua gradien A nonzero pada backward pertama: gradien A dapat nol karena B masih nol.
11. Simpan salinan CPU state adapter awal untuk membandingkan perubahan setelah update. Hindari menyalin seluruh model ke CPU demi pemeriksaan ini.

**Keluaran:** konfigurasi adapter, daftar modul, jumlah trainable, dan snapshot/hash adapter awal.

### Tahap D — Ukur loss awal dengan prosedur tetap

Hitung loss evaluasi pada keempat contoh dalam `eval()` dan tanpa gradien, adapter belum berubah. Catat loss masing-masing contoh.

Untuk ringkasan gunakan:
`sum(loss_i * jumlah_token_target_i) / sum(jumlah_token_target_i)`.

Jangan menghitung rata-rata sederhana dari empat loss yang panjang targetnya berbeda tanpa menjelaskan maknanya. Laporkan ringkasan QG dan AE juga bila berguna. Ini adalah pengukuran ulang pada **data latihan pilot**, bukan validation/test loss.

Nilai 0,6722 lama hanya pembanding historis untuk contoh QG dengan konfigurasi lama yang tercatat; jangan menjadikannya angka hardcoded atau ambang wajib yang harus persis direproduksi.

### Tahap E — Mini-training yang benar-benar memperbarui adapter

Pengaturan awal:
- batch per device = 1; gradient accumulation = 1.
- empat contoh QG–AE dijadwalkan dalam urutan yang dicatat.
- target awal delapan update optimizer yang benar-benar berhasil; bila tanpa skipped step dan satu contoh per update, setara dua putaran empat contoh.
- AdamW, learning rate 1e-4, seed 42.
- Tetapkan secara eksplisit nilai beta, epsilon, weight decay, clipping, dan scheduler yang dipilih. Contoh wajar untuk pilot: beta=(0,9;0,999), epsilon=1e-8, weight decay=0, max_grad_norm=1, scheduler konstan tanpa warmup. Semua ini pilihan pilot, bukan resep final dari paper.
- Default hanya optimizer parameter dengan `requires_grad=True`.
- Jangan mengaktifkan fused/foreach optimizer tanpa verifikasi memori dan dukungan runtime; opsi konservatif boleh dipilih secara eksplisit.

Pada jalur fp16 dengan parameter adapter fp32, implementasikan siklus AMP yang benar:
1. Kosongkan gradien.
2. Forward memakai autocast; loss jawaban dihitung dengan stabil.
3. Scale loss dan backward.
4. Unscale gradien sekali sebelum inspeksi/clipping.
5. Periksa finite dan catat norm; tangani overflow melalui mekanisme scaler, jangan menerapkan bobot dengan gradien rusak.
6. Clip gradien yang sudah di-unscale.
7. `scaler.step(optimizer)`, lalu `scaler.update()`.
8. Bedakan **percobaan langkah**, **update berhasil**, dan **update dilewati**. Jangan menggunakan return value optimizer.step() yang lazimnya None sebagai satu-satunya detektor keberhasilan.
9. Sesuaikan counter/scheduler hanya menurut kebijakan update yang benar dan terdokumentasi.
10. Batasi retry, misalnya maksimum 32 percobaan untuk target delapan update, dengan penghentian jelas bila gagal. Catat exposure per contoh ketika overflow menyebabkan retry. Jangan menjamin persis dua epoch bila ada langkah dilewati/diulang.

Loss nonfinite, overflow persisten, atau gradien kosong harus memiliki status kegagalan berbeda dari OOM. Loss tidak wajib turun pada setiap langkah karena contoh dan dropout berbeda.

**Bukti update:** bandingkan state adapter sebelum/sesudah; setidaknya satu parameter berubah, nilai finite, dan jumlah update berhasil >0. Untuk kelengkapan target delapan, laporkan apakah benar tercapai. Catat max_abs_delta/norm_delta serta perubahan B, tidak sekadar “backward selesai”.

### Tahap F — Loss akhir, penyimpanan, dan generasi sesudah latihan

1. Ulangi evaluasi loss pada empat contoh yang sama menggunakan mode, frame, prompt, dtype, dan penghitungan yang sama dengan Tahap D.
2. Bandingkan loss awal/akhir per contoh dan rata-rata berbobot token. Harapan loss menurun adalah pemeriksaan kewajaran, bukan hasil yang dijanjikan.
3. Jika loss tidak turun, laporkan apa adanya; periksa numerik, scheduler, masking, update dan data sebelum mengubah hyperparameter. Jangan menjalankan pencarian otomatis sampai muncul angka yang terlihat baik.
4. Simpan adapter ke direktori run baru di Drive. Gunakan `save_pretrained` PEFT dengan safetensors dan simpan processor/tokenizer atau identitas/revisi yang diperlukan untuk reproduksi.
5. Simpan konfigurasi resolved, daftar trainable, metadata model dasar, status training, dan log langkah.
6. Verifikasi berkas adapter/config ada, dapat dibaca, tensor finite, dan keys sesuai yang diharapkan. Keberadaan file saja belum bukti reload berhasil.
7. Simpan sebelum generasi sesudah latihan agar training tidak hilang bila generasi OOM atau runtime terputus.
8. Bersihkan gradien dan hapus optimizer/scaler yang tidak diperlukan untuk generasi; jika ingin resume, simpan dahulu state training terpisah dan jelaskan konsekuensi memori.
9. Generasi QG dengan konfigurasi identik baseline, tanpa target. Jangan mengganti prompt agar hasil sesudah latihan terlihat lebih baik.
10. Jalankan parsing/validator yang sama untuk kedua keluaran.

### Tahap G — Pemuatan ulang pada proses baru: wajib tersedia, boleh dieksekusi belakangan

Sediakan mode CLI `reload` dalam script yang sama atau entry point kecil yang terdokumentasi. Menunda eksekusinya boleh untuk membatasi sesi, tetapi **implementasinya termasuk pekerjaan ini**.

Pada proses Python baru:
1. Muat model dasar dengan identitas dan kuantisasi yang sama.
2. Pasang adapter dari direktori run dengan `PeftModel.from_pretrained` atau API setara yang terverifikasi.
3. Jangan membuat adapter acak baru lalu menyebutnya hasil reload.
4. Bandingkan state adapter termuat dengan checkpoint, dengan key mapping/toleransi numerik yang dijelaskan.
5. Hitung loss contoh yang sama dan/atau jalankan generasi QG dengan konfigurasi yang direkam. Catat keduanya jika memungkinkan.
6. Catat perbedaan hasil in-memory vs reload. Greedy tidak menjamin identitas sempurna lintas backend; jangan menutupi mismatch.
7. Jika reload belum dijalankan, laporan harus tertulis `reload_pending`, bukan selesai.

Jangan memuat dua model 9B sekaligus di GPU. Proses baru ini memang memerlukan pemuatan model kembali; klaim “model hanya dimuat sekali” berlaku pada proses training, bukan keseluruhan siklus train–reload.

## 6. Kontrak data dan loss yang tidak boleh rusak

### 6.1 Pemisahan QG dan AE

- QG: konteks + instruksi → target paket tiga soal, empat opsi, satu kunci per soal.
- AE: konteks + satu soal lengkap termasuk opsi/kunci → target penjelasan prinsip, langkah, kesimpulan, alasan tiga pengecoh, bukti, status.
- Ini dua jenis tugas pada model/adapternya yang sama. Tidak wajib dua adapter terpisah.
- QG tidak menghasilkan pembahasan; AE tidak diperlakukan sebagai target QG.
- Setiap soal/kunci dalam input AE harus konsisten dengan target AE; jangan menggabungkan contoh yang salah.

### 6.2 Masking dan alignment

Reuse `bangun_contoh` setelah diperiksa:
- input = prompt multimodal + target + penutup.
- label prompt/gambar = -100; label target = token target.
- attention mask dan tensor multimodal tetap lengkap.
- target jangan terlihat dari masa depan: model harus memakai causal masking yang benar.
- pertahankan `pixel_values`, `image_grid_thw`, dan `mm_token_type_ids` bila dibutuhkan runtime.

Untuk batch 1 dengan target kontigu di ujung, pola hemat logits pilot:
`logits = model(..., logits_to_keep=n_label + 1).logits[:, :-1]`
dan
`target = input_ids[:, -n_label:]`.

Hitung cross-entropy pada logits float32. Periksa dimensi dan shift satu posisi: prediksi token target pertama berasal dari posisi terakhir prompt; penanda akhir juga menjadi target. Jangan memperlakukan labels sebagai input generasi atau menggunakan token t untuk memprediksi dirinya sendiri.

Sediakan pemeriksaan ekuivalensi loss hemat dan loss standar pada input kecil yang muat di GPU. Bandingkan pada model/input/mode sama; hindari dua graph besar bersamaan. Jika belum bisa dijalankan lokal, sediakan langkah Colab dan tandai belum terverifikasi. Jangan mengklaim ekuivalensi numerik sudah terbukti hanya karena rumus tampak benar.

### 6.3 Independensi dan pembobotan data

Empat contoh pilot berasal dari satu segmen/sumber. Jangan membaginya acak menjadi train/test, apalagi menyebut tiga AE sebagai tiga sumber independen.

Untuk dataset penelitian nanti, gunakan `source_group_id` guna menjaga semua segmen/variasi satu sumber dalam split yang sama. Ini keputusan metodologis proyek, sejalan dengan perhatian pada pengelompokan konteks di R07.

Dengan satu contoh per langkah dan jumlah kunjungan sama, terdapat tiga langkah AE untuk setiap satu langkah QG. Catat rasio ini; jangan mengklaim objective QG–AE otomatis seimbang. Rata-rata berbobot token digunakan untuk pelaporan evaluasi, bukan alasan untuk diam-diam mengubah bobot objective training.

## 7. Validasi keluaran dan pengujian AE

Reuse `validasi_qg` dan `validasi_ae`; inspeksi keterbatasannya sebelum menyebut hasil “valid”.

Pisahkan status:
1. generation_status: berhasil, OOM, runtime error, timeout/batas waktu jika diterapkan, atau tidak dijalankan.
2. parse_status: JSON terbaca atau tidak.
3. schema_status: struktur sesuai atau tidak.
4. task_status: ok, insufficient_context, invalid_question sesuai kontrak.
5. semantic_review: belum diperiksa / diterima / perlu revisi / ditolak oleh manusia.

JSON valid dengan status `insufficient_context` adalah respons terstruktur yang mungkin sah, tetapi bukan keberhasilan menghasilkan tiga soal. Keluaran dengan `status=ok` pun harus diperiksa independen.

Simpan raw output persis sebelum:
- stripping Markdown fences;
- ekstraksi objek JSON;
- perbaikan/normalisasi;
- atau retry.

Default tidak perlu auto-repair. Bila parsing toleran disediakan, pisahkan hasil parsing ketat dan toleran, simpan transformasinya, dan jangan diam-diam memperbaiki kunci atau mengisi field hilang.

Periksa evidence terhadap **ID yang benar-benar disajikan**. Validator yang hanya mengecek seluruh manifest belum cukup untuk frame yang tidak terpilih. Keberadaan ID bukan bukti isinya mendukung klaim.

### Cakupan AE

Training empat contoh tidak membuktikan generasi AE bekerja. Sediakan opsi uji AE, bukan hanya QG:
- **Uji terkontrol:** baseline dan adapter menerima soal acuan yang sama; target pembahasan tidak diberikan.
- **Uji rangkaian:** setelah QG menghasilkan soal valid yang ditinjau, berikan soal itu ke AE. Tandai bahwa input dapat berbeda antarmodel dan kesalahan QG bisa terbawa.
- Untuk menghemat sesi, generasi AE boleh dijalankan dengan flag terpisah atau saat reload. Bila belum dijalankan, laporkan `ae_generation_pending`.

Rubrik pemeriksaan manual: soal sesuai sumber, dapat dijawab, satu kunci benar, pengecoh berbeda dan masuk akal, notasi tepat, serta AE menjelaskan kunci dan setiap pengecoh. Simpan kolom catatan untuk operator tertukar, negasi hilang/tambahan, tanda kurung salah, nilai kebenaran salah, dan aturan inferensi salah. Jangan menghitung Symbol Error Rate tanpa definisi unit dan denominator yang dibekukan terlebih dahulu.

Tidak perlu menambahkan model evaluator besar, RQUGE, ROUGE, atau penilai komersial pada mini-training. Kerangka evaluasi literatur mengarahkan review, bukan kewajiban membebani GPU pilot.

## 8. Memori, OOM, dan pergantian fase

1. Uji lama hanya mengukur maju–mundur; optimizer AdamW umumnya membuat state pada update pertama. Ukur memori setelah optimizer benar-benar bekerja, bukan setelah konstruktornya saja.
2. Catat puncak allocated dan reserved, total GPU, waktu, serta fase: load, baseline generation, forward, backward, optimizer step, evaluation, save, post generation, reload.
3. Sinkronkan GPU saat mengukur waktu. Reset peak tiap fase bila ingin membandingkan fase secara adil.
4. Cache generasi dan aktivasi training mempunyai pola memori berbeda; jangan menjanjikan inferensi pasti muat hanya karena backward muat.
5. 768 px × 9 frame adalah **cadangan kandidat**, bukan otomatis lebih hemat: token total yang pernah diukur sekitar 4.210, sedikit lebih panjang daripada 4.162 pada 1024 × 5. Memori tetap harus diukur.
6. Bila gagal, simpan konfigurasi, example_id, exception, fase, dan update berhasil. Jangan menandai “training sukses” setelah menangkap OOM.
7. Pergantian jumlah frame/resolusi, rank, optimizer, atau backend harus menghasilkan run baru. Baseline, training, dan post-generation pada perbandingan harus memakai konfigurasi data yang sama.
8. Jangan mengubah konfigurasi di tengah training lalu mencampur hasil menjadi satu kondisi eksperimen.
9. Penggunaan optimizer 8-bit/paged boleh menjadi opsi eksperimen terpisah setelah AdamW biasa terukur; beri alasan, uji kompatibilitas, dan catat perubahan. Paper QLoRA tidak berarti semua implementasi harus diam-diam mengganti optimizer.
10. Gradient checkpointing aktif saat training, cache dimatikan; saat generasi atur ulang dengan hati-hati, lalu pulihkan state saat kembali training.
11. Kernel flash-linear-attention/causal-conv1d tetap opsional. Jangan memasangnya pada baseline pilot ini tanpa uji satu faktor terpisah.
12. Jangan melakukan merge adapter ke model dasar 4-bit pada tahap ini, dan jangan menimpa folder checkpoint dasar.

## 9. Artefak run yang harus dihasilkan

Gunakan `--output-dir` yang menunjuk folder run baru, bukan folder bobot dasar. Struktur berikut adalah rancangan, boleh disesuaikan tanpa menghilangkan isinya:

| Artefak | Isi minimum |
|---|---|
| `config_resolved.json` | Semua parameter yang benar-benar dipakai, bukan hanya argumen yang ditulis pengguna |
| `environment.json` | GPU, versi paket, CUDA/cuDNN, dtype dan backend |
| `data_manifest.json` | Hash JSONL/transkrip/gambar terpilih, source IDs, urutan contoh, panjang token |
| `preflight.json` | Pemeriksaan format, waktu, evidence, masking, dan template |
| `trainable_parameters.json` | Nama modul, jumlah, dtype, layer, dan pembekuan |
| `loss_before_after.json` | Loss eval masing-masing contoh, token denominator, ringkasan berbobot |
| `train_steps.jsonl` | Satu catatan per percobaan langkah: example_id, loss, LR, scale, grad norm, update/skip, waktu, memori |
| `adapter_update_check.json` | Perubahan state adapter, finite, dan jumlah update berhasil |
| `baseline_qg.raw.txt` / `after_qg.raw.txt` | Teks generasi asli |
| `*.tokens.json` | Token baru, EOS, batas generasi dan metadata decoding |
| `*.parsed.json` / `*.validation.json` | Hasil parsing terpisah dari hasil pemeriksaan |
| `adapter/` | Adapter safetensors, config, dan informasi processor/model dasar |
| `reload_check.json` | Hasil proses baru atau status pending yang jujur |
| `summary.json` | Status setiap tahap; tidak satu boolean yang menyembunyikan kegagalan |
| `review_template.md` | Tabel ringkas untuk pemeriksaan manual QG/AE |

Tulisan log hendaknya atomik atau segera di-flush. Jika Colab terputus, run tidak boleh terlihat selesai hanya karena folder adapter sudah dibuat.

Simpan kode/config/hash revisi yang dipakai. Jika tidak ada commit baru, hash berkas yang berubah. Jangan melakukan git commit/push kecuali diizinkan instruksi pengguna/proyek. Jangan menaruh credential, token akses, atau seluruh environment variables dalam log.

## 10. CLI dan pengalaman menjalankan di Colab

Buat command sederhana, misalnya:

```bash
python scripts/latih_pilot.py --mode train --dry --model "/content/drive/MyDrive/LeveLearn/models/Qwen3.5-9B" --output-dir "/content/drive/MyDrive/LeveLearn/runs_pilot/NAMA_RUN"
```

```bash
python scripts/latih_pilot.py --mode train --model "/content/drive/MyDrive/LeveLearn/models/Qwen3.5-9B" --output-dir "/content/drive/MyDrive/LeveLearn/runs_pilot/NAMA_RUN"
```

```bash
python scripts/latih_pilot.py --mode reload --run-dir "/content/drive/MyDrive/LeveLearn/runs_pilot/NAMA_RUN"
```

Ini **kontrak CLI yang perlu diimplementasikan**, bukan perintah yang sudah dapat dijalankan pada kode lama.

Flag minimum yang berguna: model, JSONL, output/run directory, mode, dry, jumlah frame, resolusi, crop, steps, LR, r/alpha/dropout, seed, batas token generasi, batas retry, dan opsi generasi AE. Jangan memperkenalkan terlalu banyak pilihan yang belum diuji.

Panduan Colab harus mencakup:
1. Memilih GPU dan memeriksa kapasitas.
2. Mount Drive dan memeriksa folder model.
3. Upload/ekstraksi ZIP baru yang memuat script, helper proyek, dan data.
4. Menormalisasi kapitalisasi folder.
5. Instalasi dependensi yang terkontrol, tidak mengubah torch/torchvision bawaan tanpa alasan.
6. Restart hanya jika memang dibutuhkan, dengan urutan sel yang jelas.
7. Dry-run, train, lalu reload dalam proses baru.
8. Lokasi log dan file yang perlu dikirim kembali.

Jangan memberi perintah destruktif untuk menghapus seluruh Drive atau folder model. Jangan memuat model lokal laptop yang kapasitas GPU-nya tidak cukup; laptop dapat menjalankan pemeriksaan non-GPU.

## 11. Pemeriksaan penerimaan

Pisahkan kelulusan teknik menjadi tiga milestone:

### M1 — Update dan simpan
- Empat contoh lolos preflight atau isu datanya diselesaikan secara terdokumentasi.
- Hanya adapter target yang trainable.
- Ada update optimizer nyata, jumlah update berhasil diketahui, parameter adapter berubah dan finite.
- Loss awal/akhir dapat dibandingkan dengan cara yang sama, terlepas apakah sudah menurun.
- Adapter tersimpan dan dapat dibaca.

### M2 — Generasi dan validasi
- Baseline dan post QG dijalankan tanpa target dengan konfigurasi sebanding, atau kegagalannya dilaporkan.
- Raw output, token, parsing, dan hasil validator terpisah.
- Tidak ada klaim “soal benar” hanya dari JSON/schema valid.
- Generasi AE dicoba atau ditandai pending; training AE tidak disamakan dengan pengujian AE.

### M3 — Persistensi
- Proses baru benar-benar memuat adapter tersimpan, bukan adapter baru.
- State dan keluaran/loss diperiksa dengan identitas model dasar yang sama.
- Jika belum dijalankan, milestone belum selesai.

Verifikasi kode yang perlu dilakukan:
- Syntax/import dan CLI help.
- Pemeriksaan tidak ada target dalam input generasi.
- Masking dan shift jawaban, termasuk penutup.
- Ekuivalensi loss hemat vs standar pada input kecil.
- Unit yang bermakna untuk invalid JSON, evidence tak disajikan, pemetaan AE salah, dan jalur deteksi skipped update.
- Satu uji GPU nyata mencakup optimizer state; selanjutnya uji reload pada proses baru.

Jangan menjalankan training palsu atau mengisi log dengan angka perkiraan. Bila GPU tidak tersedia, selesaikan implementasi dan pemeriksaan yang bisa dijalankan, lalu tulis tepat bagian mana yang masih menunggu Colab.

## 12. Bentuk jawaban Claude setelah selesai mengimplementasikan

Sampaikan dalam bahasa Indonesia:
1. Ringkasan perubahan kode dan alasan.
2. Berkas yang dibuat/diubah.
3. Tabel keputusan: sumber/ADR → penerapan → batas bukti.
4. Pemeriksaan yang benar-benar dijalankan beserta hasil.
5. Yang belum diuji di GPU.
6. Sel Colab siap salin sesuai CLI aktual.
7. Keluaran apa yang harus diperiksa setelah run.
8. Batas kesimpulan: empat contoh satu sumber hanya membuktikan mekanik.

Jangan menyebut “model sudah paham”, “berhasil membuat soal” sebelum melihat generasi, “0,6722 adalah kemiripan”, atau “pengaturan ini sudah paling baik menurut jurnal”.

## 13. Ringkasan alasan yang dapat dipakai saat bimbingan

Kami menggunakan QLoRA sebagai pendekatan adaptasi hemat memori dengan dasar model 4-bit dan parameter tambahan LoRA. Contoh latih memasangkan frame, transkrip, dan instruksi dengan soal atau pembahasan acuan yang telah diperiksa. Mini-training memeriksa pembaruan parameter dan ketahanan alur penyimpanan serta generasi. Kualitas belum disimpulkan dari loss pada contoh latihan; penilaian selanjutnya membutuhkan sumber video terpisah dan pemeriksaan isi soal, kunci, pengecoh, notasi, serta pembahasan. Jumlah frame, rank, learning rate, dan jumlah langkah pada pilot merupakan konfigurasi awal yang diuji, bukan hasil optimasi final.

## 14. Daftar sumber inti yang dapat dibuka Claude

- **R01 — [LORA_ LOW-RANK ADAPTATION OF LARGE LAN- GUAGE MODELS](https://drive.google.com/file/d/1xG8xpzDhj1Pn1vSCvMnoTJ1QalG-cUAK/view)**
- **R02 — [QLORA_ Efficient Finetuning of Quantized LLMs](https://drive.google.com/file/d/1EdYZfYRatijEOWwK6DF3QpCa0i1nFZU1/view)**
- **R03 — [Efficient fine‐tuning of vision‐language adapters in chemical VLMs for molecular image‐text tasks](https://drive.google.com/file/d/1QQXk4myYAJ196zTGf53sdFyrDc8icC9-/view)**
- **R04 — [Enhancing the Learning Experience_ Using Vision-Language Models to Generate Questions  for Educational Videos](https://drive.google.com/file/d/1yUsoppG_bfcbipoc9m0jtOq2lImByrcw/view)**
- **R05 — [QuizTools_ Question Generation with Explanations](https://drive.google.com/file/d/1IeJ3g0vYghayjsgTIssY2dUtGSvakAG_/view)**
- **R06 — [Context Selection and Rewriting for Video-based Educational Question Generation](https://drive.google.com/file/d/1uZ373O-Fne6AiEtdy6RBtyxpUMHzINXX/view)**
- **R07 — [From Generation to Evaluation_ A Resource for Error-Categorized  Question Generation from Video Transcripts](https://drive.google.com/file/d/1PtE3ZEH00awrO_S4IQoGLdiQg_i5bZOL/view)**
- **R08 — [QGEval_ Benchmarking Multi-dimensional Evaluation for  Question Generation](https://drive.google.com/file/d/1yd8auNSDPii3yWTsKJqGwf5nfvOpmdVH/view)**
- **R09 — [RQUGE_ Reference-Free Metric for Evaluating Question Generation by  Answering the Question](https://drive.google.com/file/d/1-z4lEHIRR09SDav5qYvi3ZM5_ryJDuxy/view)**
- **R10 — [Assessing the Quality of Multiple-Choice Questions Using  GPT-4 and Rule-Based Methods](https://drive.google.com/file/d/1o_eVUEWlMWDVVQx5otGTbmH4pW3Ynne7/view)**
- **R11 — [Automatic Distractor Generation in Multiple-Choice Questions Using Large Language Models with Expert-Informed  Distractor Strategies](https://drive.google.com/file/d/1-YWjYBi57iLs2PHAFrEj_6kAVZmuBmO-/view)**
- **R12 — [Video-MME_ The First-Ever Comprehensive Evaluation Benchmark  of Multi-modal LLMs in Video Analysis](https://drive.google.com/file/d/1pwUOXAonUzdLPvOscE7gowAyHFODLP_p/view)**
- **R13 — [OmniEdu_ Open Foundation Models for Learning and Teaching](https://drive.google.com/file/d/1l1Db4OO6HUIDNlTBiFUQi0wVsewXfzJQ/view)**
- **R14 — [DoRA_ Weight-Decomposed Low-Rank Adaptation](https://drive.google.com/file/d/19W8jp1nWANN3w2jopgnvZ1Emcrn45TQY/view)**
- **R15 — [LoftQ_ LoRA-Fine-Tuning-Aware Quantization for Large  Language Models](https://drive.google.com/file/d/1h9QRDUbE1tzEozy-fZtgmkfnfYTV2q_E/view)**
- **R16 — [PiSSA_ Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://drive.google.com/file/d/1NHISr0B4lv9bWoBQ_z_wkXna5qqcitYl/view)**
- **R17 — [ADALORA_ ADAPTIVE BUDGET ALLOCATION FOR PARAMETER-EFFICIENT FINE-TUNING](https://drive.google.com/file/d/1c-5n92gEinyUeY7k46fwsbAHGQcMym6B/view)**

Identitas primer tambahan yang diperiksa:
- R03: https://doi.org/10.1186/s13321-026-01214-5 ; halaman penerbit https://link.springer.com/article/10.1186/s13321-026-01214-5 .
- R05: https://doi.org/10.3390/a19100858 ; halaman penerbit https://www.mdpi.com/1999-4893/19/10/858 .

## Lampiran — Inventaris metadata folder

Daftar ini merekam 165 berkas yang ditampilkan folder pada 3 November 2026. Label **teks diakses** berarti PDF tersedia sebagai teks dan bagian relevannya ditelaah; bukan klaim setiap paragraf dibaca. **Metadata saja** berarti belum dipakai untuk klaim isi. Nama berkas dipertahankan, bukan metadata publikasi yang sudah diverifikasi. Nama mirip dapat merupakan duplikat; nama generik seperti `paper.pdf` tidak dapat diidentifikasi dari nama saja. Tidak ada sumber tambahan yang diambil dari hasil pencarian Drive kosong sebagai pengganti isi folder.

| No. | Berkas | Cakupan pemeriksaan |
|---|---|---|
| 1 | [A Systematic Literature Review of  Retrieval-Augmented Generation_ Techniques,  Metrics, and Challenges.pdf](https://drive.google.com/file/d/1ZrZYUiP9oM5KlqysFRiWy72HioQgN4t-/view) | Metadata saja |
| 2 | [A Slide Annotation System with Multimodal Analysis for Video Presentation Review.pdf](https://drive.google.com/file/d/1bscFbF_bLAyci48bVVpr5BI6s25DjfBy/view) | Metadata saja |
| 3 | [∞-VIDEO A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation.pdf](https://drive.google.com/file/d/1FRjs9rCKYjfH9K0f0HYuXupO5uY5YwfU/view) | Metadata saja |
| 4 | [Visual Content Detection in Educational Videos with Transfer Learning and Dataset Enrichment.pdf](https://drive.google.com/file/d/1exU0AWtpe9uG8noAFITiU7XOMJkyeD4L/view) | Metadata saja |
| 5 | [Video-MME_ The First-Ever Comprehensive Evaluation Benchmark  of Multi-modal LLMs in Video Analysis.pdf](https://drive.google.com/file/d/1pwUOXAonUzdLPvOscE7gowAyHFODLP_p/view) | Teks diakses, R12 |
| 6 | [Video-LLaMA An Instruction-tuned Audio-Visual Language Model for Video Understanding.pdf](https://drive.google.com/file/d/1HbP1FMqfxLxBsDVWpLehUTxbS1Ldkcga/view) | Metadata saja |
| 7 | [Unsupervised Audio-Visual Lecture Segmentation.pdf](https://drive.google.com/file/d/1CIjoSLne_8nnWrR0PrnLKBeowmSt3YWC/view) | Metadata saja |
| 8 | [Trend and Issues in Educational Video Research_ A Bibliometric Mapping Analysis.pdf](https://drive.google.com/file/d/1C0ha0XAf4DjYUUli_8llRCFqOg_vinQo/view) | Metadata saja |
| 9 | [TimeLens Rethinking Video Temporal Grounding with Multimodal LLMs.pdf](https://drive.google.com/file/d/1nXcTRU_OM61OW5DX8s911ypswNXtVL7j/view) | Metadata saja |
| 10 | [Towards automatic question generation using pre-trained model in academic field for Bahasa Indonesia.pdf](https://drive.google.com/file/d/1MmXS1mqZAJyc10WqL2mm2eM4KuPQtuou/view) | Metadata saja |
| 11 | [the-inclusive-glossary-an-embedded-interactive-approach-to-accessible-and-inclusive-learning.pdf](https://drive.google.com/file/d/1JsWAZnNq88tlZlgcRGzYGFKfy8z780ym/view) | Metadata saja |
| 12 | [Test of Time Instilling Video-Language Models with a Sense of Time.pdf](https://drive.google.com/file/d/1FM_Mk9klQEZuxsDa40lnizeawnPIOnNf/view) | Metadata saja |
| 13 | [SEASON Mitigating Temporal Hallucination in Video Large Language Models via Self-Diagnostic Contrastive Decoding.pdf](https://drive.google.com/file/d/1zJEQiPqlfZTCs0s_brk6CdxmJMEldYHt/view) | Metadata saja |
| 14 | [RAG in Self-Regulated Personalised Learning in Digital Education.pdf](https://drive.google.com/file/d/1juAUW3zDaiiNTvwk20IONY2BTFwFA3Cn/view) | Metadata saja |
| 15 | [On Formal Methods Thinking in Computer Science Education.pdf](https://drive.google.com/file/d/1sLSwQwMs5jNmSdcp2GbXD_ZY5l9TSZfF/view) | Metadata saja |
| 16 | [Multimedia design for learning—An overview of reviews with meta-meta-analysis.pdf](https://drive.google.com/file/d/19MyZvOlVH_i899WD4Ou0C6zCjGwqgHJ6/view) | Metadata saja |
| 17 | [MMReason_ An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI.pdf](https://drive.google.com/file/d/1_NkDDDXQWHfddcO0IWr96BW_77fyLgep/view) | Metadata saja |
| 18 | [MATHVISTA_ EVALUATING MATHEMATICAL REASON- ING OF FOUNDATION MODELS IN VISUAL CONTEXTS.pdf](https://drive.google.com/file/d/1kAGwGPBmbN25U40mW7ZhSGszUQLmqzO2/view) | Metadata saja |
| 19 | [Learning in double time_ The effect of lecture video speed on immediate and delayed comprehension.pdf](https://drive.google.com/file/d/1nRtR-zPygQPaKZA0w14T9zSg201eDKA6/view) | Metadata saja |
| 20 | [Human-in-the-Loop Machine Learning to Increase Video Accessibility for Visually Impaired and Blind Users.pdf](https://drive.google.com/file/d/17SKsKTtA-VgqcJeNJek43XOWsl3bJYjw/view) | Metadata saja |
| 21 | [Grounded-VideoLLM Sharpening Fine-grained Temporal Grounding in Video Large Language Models.pdf](https://drive.google.com/file/d/1yybpnL7t8gYgxLZdzIH2_TGnY8E3M8RN/view) | Metadata saja |
| 22 | [Generation and Use of Hints and Feedback in a Hilbert-Style Axiomatic Proof Tutor.pdf](https://drive.google.com/file/d/10FfI5Bb0ufW27A2boAo7dYTD1hcomnno/view) | Metadata saja |
| 23 | [Formative and Summative Automated Assessment with Multiple- Choice Question Banks.pdf](https://drive.google.com/file/d/1ouTrCnH32OopWzZK1mPnYpd1JTWyRcBs/view) | Metadata saja |
| 24 | [Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation.pdf](https://drive.google.com/file/d/11Pey2lf1fUS_25DQ0j81Ls85bbgwz-Kc/view) | Metadata saja |
| 25 | [Does watching an explainer video help learning with subsequent text_ – Only when prompt-questions are provided.pdf](https://drive.google.com/file/d/1Wi_yjFnpX2Zr8WUztStd9zvfXcbqvBi_/view) | Metadata saja |
| 26 | [Describing the What and Why of Students’ Difficulties in Boolean Logic.pdf](https://drive.google.com/file/d/1VR8ybGOl07qH_-2thOa4BoppqnCYmMGP/view) | Metadata saja |
| 27 | [Connecting Computing Students’ External Help Resource Preferences and Internal Help Resource Usage_ 2021-2025.pdf](https://drive.google.com/file/d/1k6291BI81z3XahErqon5nL4w2D4O7jp8/view) | Metadata saja |
| 28 | [Automated Whiteboard Lecture Video  Summarization by Content Region Detection and  Representation.pdf](https://drive.google.com/file/d/1S7iK5XdCL8BB2jUOY6RSs-cTmty6eQ3e/view) | Metadata saja |
| 29 | [AI tutoring outperforms in-class active learning_ an RCT introducing a novel research-based design in an authentic educational setting.pdf](https://drive.google.com/file/d/1MaWYlR-WPzpDO_-NR0dpAcd5rJvJWDqp/view) | Metadata saja |
| 30 | [When Video Improves Learning in Higher Education.pdf](https://drive.google.com/file/d/1v8pbB_7Usqpeb93CXKvN-vuUz16RRenA/view) | Metadata saja |
| 31 | [The Role of Logic in Teaching Proof.pdf](https://drive.google.com/file/d/1MhD_ZbVgyfnIMoOmXM0FY9g62tdxiv6j/view) | Metadata saja |
| 32 | [The effects of video lecture viewing strategies on cognitive load.pdf](https://drive.google.com/file/d/1F9yVsxqkCSXlxqFzzSSAwR7Ug86crqFB/view) | Metadata saja |
| 33 | [Students’ propositional logic thinking in higher education from the perspective of disciplines.pdf](https://drive.google.com/file/d/1GXtM4I-Xcuq-yWfUHhPVrFXLE57CB76l/view) | Metadata saja |
| 34 | [STUDY LITERATUR PEMAHAMAN MAHASISWA TENTANG LOGIKA  INFORMATIKA DALAM PEMROGRAMAN.pdf](https://drive.google.com/file/d/1InSu7pmvas0MblmmYFN87Tn8yzE7slCb/view) | Metadata saja |
| 35 | [NADIA - Natural DeductIon proof Assistant.pdf](https://drive.google.com/file/d/1xfP1ma3Z2zH-1QbRUS3w1-uZAbo08TzI/view) | Metadata saja |
| 36 | [NADIA - Natural DeductIon proof Assistant_ Um Assistente de Provas de Dedução Natural para Lógica Proposicional e Lógica  de Pre.pdf](https://drive.google.com/file/d/1MF3osX8gbswNfybeKxrB1foMZLAVCsP7/view) | Metadata saja |
| 37 | [Instructors’ perceptions of a large language model framework for generating automated multiple‐choice questions from video transcripts.pdf](https://drive.google.com/file/d/1i779joOuvasdVMt0sbhQnG00pORamYsf/view) | Metadata saja |
| 38 | [Improving mathematical proof based on computational thinking components for prospective teachers in abstract algebra courses.pdf](https://drive.google.com/file/d/1fWacSd0IJ6_JIjEplj3EfcqDzeK9tvED/view) | Metadata saja |
| 39 | [From Generation to Evaluation_ A Resource for Error-Categorized  Question Generation from Video Transcripts.pdf](https://drive.google.com/file/d/1PtE3ZEH00awrO_S4IQoGLdiQg_i5bZOL/view) | Teks diakses, R07 |
| 40 | [Formal Methods in Industry.pdf](https://drive.google.com/file/d/1gCQH5KL35OXtCj4hmIVJmA0YSOMtFBCx/view) | Metadata saja |
| 41 | [Enhancing Video Vision Language Model with Hippocampal Sensing.pdf](https://drive.google.com/file/d/1B3DILnUWPw84ylY_04vqk3Y1orzhWW3L/view) | Metadata saja |
| 42 | [Deep learning approaches to lexical simplification_ A survey.pdf](https://drive.google.com/file/d/1-NyPv7QdnlfHRvrRiGk1RTvjQaOoNAVT/view) | Metadata saja |
| 43 | [Enhancing the Learning Experience_ Using Vision-Language Models to Generate Questions  for Educational Videos.pdf](https://drive.google.com/file/d/1yUsoppG_bfcbipoc9m0jtOq2lImByrcw/view) | Teks diakses, R04 |
| 44 | [Comparison of Natural Deduction Theorem Provers used in Electronic Tutoring Systems.pdf](https://drive.google.com/file/d/1G-Zv4AfwqkwTgR5dDc8SZSOuoB_Jh9HB/view) | Metadata saja |
| 45 | [Assessing the Quality of Multiple-Choice Questions Using  GPT-4 and Rule-Based Methods.pdf](https://drive.google.com/file/d/1o_eVUEWlMWDVVQx5otGTbmH4pW3Ynne7/view) | Teks diakses, R10 |
| 46 | [Context Selection and Rewriting for Video-based Educational Question Generation.pdf](https://drive.google.com/file/d/1uZ373O-Fne6AiEtdy6RBtyxpUMHzINXX/view) | Teks diakses, R06 |
| 47 | [DoRA_ Weight-Decomposed Low-Rank Adaptation.pdf](https://drive.google.com/file/d/19W8jp1nWANN3w2jopgnvZ1Emcrn45TQY/view) | Teks diakses, R14 |
| 48 | [LORA_ LOW-RANK ADAPTATION OF LARGE LAN- GUAGE MODELS.pdf](https://drive.google.com/file/d/1xG8xpzDhj1Pn1vSCvMnoTJ1QalG-cUAK/view) | Teks diakses, R01 |
| 49 | [PiSSA_ Principal Singular Values and Singular Vectors Adaptation of Large Language Models.pdf](https://drive.google.com/file/d/1NHISr0B4lv9bWoBQ_z_wkXna5qqcitYl/view) | Teks diakses, R16 |
| 50 | [Efficient fine‐tuning of vision‐language adapters in chemical VLMs for molecular image‐text tasks.pdf](https://drive.google.com/file/d/1QQXk4myYAJ196zTGf53sdFyrDc8icC9-/view) | Teks diakses, R03 |
| 51 | [LoftQ_ LoRA-Fine-Tuning-Aware Quantization for Large  Language Models.pdf](https://drive.google.com/file/d/1h9QRDUbE1tzEozy-fZtgmkfnfYTV2q_E/view) | Teks diakses, R15 |
| 52 | [ADALORA_ ADAPTIVE BUDGET ALLOCATION FOR PARAMETER-EFFICIENT FINE-TUNING.pdf](https://drive.google.com/file/d/1c-5n92gEinyUeY7k46fwsbAHGQcMym6B/view) | Teks diakses, R17 |
| 53 | [QLORA_ Efficient Finetuning of Quantized LLMs.pdf](https://drive.google.com/file/d/1EdYZfYRatijEOWwK6DF3QpCa0i1nFZU1/view) | Teks diakses, R02 |
| 54 | [GATED DELTA NETWORKS _  IMPROVING  MAMBA  2 WITH  DELTA RULE.pdf](https://drive.google.com/file/d/1YGJ_bXnqetHeRLb9A6xRnoOkmHjGvwtq/view) | Metadata saja |
| 55 | [OmniEdu_ Open Foundation Models for Learning and Teaching.pdf](https://drive.google.com/file/d/1l1Db4OO6HUIDNlTBiFUQi0wVsewXfzJQ/view) | Teks diakses, R13 |
| 56 | [Effects of self-explaining feedback on learning from problem-solving errors.pdf](https://drive.google.com/file/d/15cUtEHvaAr8iI1jJc-VJhDynW6NWXNth/view) | Metadata saja |
| 57 | [Evaluation of an online text simplification editor using manual and automated metrics for perceived and actual text difficulty.pdf](https://drive.google.com/file/d/1_TxEgq78627fmmvdftSlpHeBrYduJ-kQ/view) | Metadata saja |
| 58 | [The effect of worked examples on learning solution steps and knowledge transfer.pdf](https://drive.google.com/file/d/1LYb2Jr7btGQfUYgTe3mfpaW38gJ5_cYY/view) | Metadata saja |
| 59 | [Do Text Simplification Systems Preserve Meaning_ A Human Evaluation via Reading Comprehension.pdf](https://drive.google.com/file/d/1lwdehXnJsvS9dM0UioZ2s3MaDCJnxfB1/view) | Metadata saja |
| 60 | [What are explanatory proofs in mathematics and how can they contribute to teaching and learning_.pdf](https://drive.google.com/file/d/1XCq4TX2EurAKf5EnvJfGA7T6I57WWNvE/view) | Metadata saja |
| 61 | [Facilitating Transfer through Incorrect Examples and  Explanatory Feedback.pdf](https://drive.google.com/file/d/15gwLqK-p_8L-_HSMbHUFeevF3ScAbbTA/view) | Metadata saja |
| 62 | [From belief to evidence_ simpler immediate feedback improves language learning and confidence in semi-open-ended questions.pdf](https://drive.google.com/file/d/1mkdoFJMo-WFz8p6MRu69nkn-PIyAbqZ3/view) | Metadata saja |
| 63 | [The Power of Feedback Revisited_ A Meta-Analysis of Educational Feedback Research.pdf](https://drive.google.com/file/d/1jv3aMSCDVUSkyV0S5AMDOnXeRrLa6M5d/view) | Metadata saja |
| 64 | [Psychometric performance and student perceptions of AI- versus student-generated multiple-choice questions_ a single-center randomized controlled trial.pdf](https://drive.google.com/file/d/1LjeKG_Sg95mNGKEIk8mSLZo1eDtxLFfj/view) | Metadata saja |
| 65 | [FACTSCORE_ Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.pdf](https://drive.google.com/file/d/1l2L8Z2_Av68tpNsBUJbTU6CvM9CGstlG/view) | Metadata saja |
| 66 | [RQUGE_ Reference-Free Metric for Evaluating Question Generation by  Answering the Question.pdf](https://drive.google.com/file/d/1-z4lEHIRR09SDav5qYvi3ZM5_ryJDuxy/view) | Teks diakses, R09 |
| 67 | [MM-MATH_ Advancing Multimodal Math Evaluation with Process Evaluation and Fine-grained Classification.pdf](https://drive.google.com/file/d/1l5r1Ao3C1zfSFS5g5AQ6D3ubAaZgfW7d/view) | Metadata saja |
| 68 | [Instruction-Tuned Large-Language Models for Quality Control in Automatic Item Generation_ A Feasibility Study.pdf](https://drive.google.com/file/d/1wXuub1AitH4ZydK4uy6htm0dBwKFtjUh/view) | Metadata saja |
| 69 | [Transformer and Large Language Models for Automatic Multiple-Choice Question Generation_ A Systematic Literature Review.pdf](https://drive.google.com/file/d/1xq_YJ1IdgtfzsElA_h6dK6yfg9P9GUOK/view) | Metadata saja |
| 70 | [Automatic distractor generation in multiple-choice questions_ a systematic literature review.pdf](https://drive.google.com/file/d/1EPhijLPfJ8ZF1yPqETnyJL39iYJgBEZb/view) | Metadata saja |
| 71 | [A Systematic Review of Automatic Question Generation for Educational Purposes.pdf](https://drive.google.com/file/d/1Y9PfACwmmZUsj0tJr2n1dBfjwSjwb0V1/view) | Metadata saja |
| 72 | [A literature review of research on question generation in education.pdf](https://drive.google.com/file/d/108DKgmsvcp4Szr0U9URScGumtlDacfA3/view) | Metadata saja |
| 73 | [Automating question generation from  educational text.pdf](https://drive.google.com/file/d/1sgJjzggKjUHbkJlqCGcwkB59Lz04lNeI/view) | Metadata saja |
| 74 | [Analysis of LLMs for educational question classification and generation.pdf](https://drive.google.com/file/d/1xYb4GWWQQ8BK0kPsjAMH_KClSqRKNxEW/view) | Metadata saja |
| 75 | [Automated Educational Question Generation at Different Bloom’s Skill Levels using Large Language Models_ Strategies and Evaluation.pdf](https://drive.google.com/file/d/1ufriajgdpkmyHy0KfRRPxaLPESTt3Luv/view) | Metadata saja |
| 76 | [Advancing AI in Higher Education_ A Comparative Study of Large Language Model-Based Agents for Exam Question Generation, Improvement, and Evaluation.pdf](https://drive.google.com/file/d/1qbaTTp2XVj4Djz8zhYRpN_VfoU_z9xUs/view) | Metadata saja |
| 77 | [MCQGen_ A Large Language Model-Driven MCQ Generator for Personalized Learning.pdf](https://drive.google.com/file/d/1jCRHXS9OHKw0OthnO7Of4wrgmMRhaVrV/view) | Metadata saja |
| 78 | [Can large language models meet the challenge of generating school-level questions_.pdf](https://drive.google.com/file/d/13pfWDlhkRtPXral6tJAfdsveTTsQonsw/view) | Metadata saja |
| 79 | [QuizTools_ Question Generation with Explanations.pdf](https://drive.google.com/file/d/1IeJ3g0vYghayjsgTIssY2dUtGSvakAG_/view) | Teks diakses, R05 |
| 80 | [Automatic Distractor Generation in Multiple-Choice Questions Using Large Language Models with Expert-Informed  Distractor Strategies.pdf](https://drive.google.com/file/d/1-YWjYBi57iLs2PHAFrEj_6kAVZmuBmO-/view) | Teks diakses, R11 |
| 81 | [EduQG_ A Multi-Format Multiple-Choice Dataset for the Educational Domain.pdf](https://drive.google.com/file/d/1R9__-h63_3ZPS-vrKtCMivDxUBsTCdEu/view) | Metadata saja |
| 82 | [ExamQ-Gen_ Instructor-in-the-Loop Generation of Self-Contained Exam Questions from Course Materials and Decision-Support Grading.pdf](https://drive.google.com/file/d/1j82Scntv1-TMKY8PufhqxWz9fzdyWVYp/view) | Metadata saja |
| 83 | [Generation and Assessment of Multiple-Choice Questions from  Video Transcripts using Large Language Models.pdf](https://drive.google.com/file/d/1JPo6xRLV8Y5T56-PHixmPTWmpndhQF5A/view) | Metadata saja |
| 84 | [Cognitively Diverse Multiple-Choice Question Generation_ A Hybrid Multi-Agent Framework with Large Language Models.pdf](https://drive.google.com/file/d/1JplJYlZk7J-Dkf15bew1HOMtzJNSmX0q/view) | Metadata saja |
| 85 | [Video Tutorials in the Traditional Classroom_ The Effects on Different Types of Cognitive Load.pdf](https://drive.google.com/file/d/1LZ_CIb10QObPKDHPp4fym_RazOX8OU5Q/view) | Metadata saja |
| 86 | [Automatic, Meta and Human Evaluation for Multimodal Summarization  with Multimodal Output.pdf](https://drive.google.com/file/d/1s_AdAweg0CcZ90U8lvM5vPS6Vq9fM0uD/view) | Metadata saja |
| 87 | [The effectiveness of enhanced interaction features in educational videos_ a meta-analysis.pdf](https://drive.google.com/file/d/1xkwgcIFc_LgHjD0GE3OWkx7yB3A-uyHT/view) | Metadata saja |
| 88 | [INQUIRER_ Harnessing internal knowledge graphs for video question generation.pdf](https://drive.google.com/file/d/1I_5eiegl-MedgTh_8fbCs6I-2A0qz1lF/view) | Metadata saja |
| 89 | [SummEval_ Re-evaluating Summarization Evaluation.pdf](https://drive.google.com/file/d/1h4H79JgWu64ZVQu1H54wNrWOSAdktvZw/view) | Metadata saja |
| 90 | [Generative learning activities for online multimedia learning_ when summarizing is effective but drawing is not.pdf](https://drive.google.com/file/d/1yU-zW8ZbUXahixRJienWEd-ixvm4bOcX/view) | Metadata saja |
| 91 | [Learning to Ask_ Multi-Decoder Fine-Tuning for Multi-Hop Visual  Question Generation with External Knowledge.pdf](https://drive.google.com/file/d/1t9Otfwj7hSKG0p7VUIhm3tfrWlzTR_IU/view) | Metadata saja |
| 92 | [Distractor Generation in Multiple-Choice Tasks_ A Survey of Methods,  Datasets, and Evaluation.pdf](https://drive.google.com/file/d/1Kd3zcdqNUetmX3-1sdbUFFGXgc-8tjAR/view) | Metadata saja |
| 93 | [Chain-of-Exemplar_ Enhancing Distractor Generation for Multimodal  Educational Question Generation(1).pdf](https://drive.google.com/file/d/1YWPVTlM-FQfCGsBULEWU_TfrNa5jN6t8/view) | Metadata saja |
| 94 | [Learning from academic video with subtitles_ When foreign language proficiency matters.pdf](https://drive.google.com/file/d/1VNDjRDmzE8tR0Jx2nmGD5KFRJzR9rELl/view) | Metadata saja |
| 95 | [Focused self‐explanation prompts and segmenting foster pre‐service teachers’ professional vision ‐ but only during training!.pdf](https://drive.google.com/file/d/15viH51fPBoXSD2P2A1boMYlkGN-vtOrn/view) | Metadata saja |
| 96 | [Language Learning - 2024 - Shi - The Effects of Task Repetition on the Processing and Acquisition of Technical Vocabulary.pdf](https://drive.google.com/file/d/1IwW8-WJnixyu4ZHiH7iOz440lktxCbfz/view) | Metadata saja |
| 97 | [Generating Questions, Answers, and Distractors for Videos_ Exploring Semantic Uncertainty of Object Motions.pdf](https://drive.google.com/file/d/11g9IloGX1_Y1RAusD0RWr6HflJ35d0B9/view) | Metadata saja |
| 98 | [Towards Process-Oriented, Modular, and Versatile Question Generation  that Meets Educational Needs.pdf](https://drive.google.com/file/d/1OU5JHMLb5cttu2gFLPCFzWv_Ak9Nqcp0/view) | Metadata saja |
| 99 | [From Objectives to Questions_ A Planning-based Framework for  Educational Mathematical Question Generation.pdf](https://drive.google.com/file/d/1ehaaT8hz4njiDAcmfi8DxiJIUKVlzXet/view) | Metadata saja |
| 100 | [Visual Transcripts_ Lecture Notes from Blackboard-Style Lecture Videos.pdf](https://drive.google.com/file/d/1vk6Ze_oD6MpsfdcqAjqlNsKdmjFIRTUU/view) | Metadata saja |
| 101 | [What Is That Talk About_ A Video-to-Text Summarization Dataset for  Scientific Presentations.pdf](https://drive.google.com/file/d/1mPqfQLYGepwnARErMIzBXQC8n9sKozr0/view) | Metadata saja |
| 102 | [NExT-QA_ Next Phase of Question-Answering to Explaining Temporal Actions.pdf](https://drive.google.com/file/d/1WeVOrMPG1XKTAjHyFvVWlBNA6gSeZIk-/view) | Metadata saja |
| 103 | [How do enhanced videos support generative learning and conceptual understanding in individuals and groups_.pdf](https://drive.google.com/file/d/19mVb2NumyusO4bGt_s5CBvWuU6N0VCzr/view) | Metadata saja |
| 104 | [Educational Question Generation of Children Storybooks via Question Type Distribution Learning and Event-Centric Summarization.pdf](https://drive.google.com/file/d/1qNag5cCmS7KLfmnxbDb_RYqBIwGPtaEs/view) | Metadata saja |
| 105 | [Signaling cues and focused prompts for professional vision support_ The interplay of instructional design and situational interest in preservice teachers’ video analysis.pdf](https://drive.google.com/file/d/15FjMWfMbsIUcOIM1gNJ0b3L3isk3Qu4V/view) | Metadata saja |
| 106 | [QGEval_ Benchmarking Multi-dimensional Evaluation for  Question Generation.pdf](https://drive.google.com/file/d/1yd8auNSDPii3yWTsKJqGwf5nfvOpmdVH/view) | Teks diakses, R08 |
| 107 | [Asking and Answering Questions to Evaluate the Factual Consistency of  Summaries.pdf](https://drive.google.com/file/d/1000CrqG0ltqOANshOIMDkK5cjrScbJ1X/view) | Metadata saja |
| 108 | [Detecting and Mitigating Challenges in Zero-Shot Video Summarization with Video LLMs.pdf](https://drive.google.com/file/d/1cEguvUI2sP5sZbvO99sSn5iW247d7Jo-/view) | Metadata saja |
| 109 | [Can prompts improve self‐explaining an online video lecture_ Yes, but do not disturb!.pdf](https://drive.google.com/file/d/1DNvq543vw-o_nWsOs-vxLSx26e6L-5XV/view) | Metadata saja |
| 110 | [Covering Uncommon Ground_  Gap-Focused Question Generation for Answer Assessment.pdf](https://drive.google.com/file/d/1dEUd97kdATie_ffPH4m0h__Rw-fyBc85/view) | Metadata saja |
| 111 | [Eye movement patterns differ while watching captioned videos of second language vs. mathematics lessons.pdf](https://drive.google.com/file/d/13t4QNqAcpqE37ZkM-gYk28JCYq0RvF5W/view) | Metadata saja |
| 112 | [User Interaction Behavior Analysis for Cognitive Load Detection in Online Learning Processes.pdf](https://drive.google.com/file/d/1T9W7QwBL1y7jsGmlPe_nSFBzyQiNSioj/view) | Metadata saja |
| 113 | [ECIS-VQG_ Generation of Entity-centric Information-seeking Questions  from Videos.pdf](https://drive.google.com/file/d/1iuw3hVACFXSc2UWkQpjbhbtXCfL3nuoj/view) | Metadata saja |
| 114 | [Distractor Generation based on Text2Text Language Models with Pseudo  Kullback-Leibler Divergence Regulation.pdf](https://drive.google.com/file/d/1gV3TNXBmRf9XK5B0v5y3YAj_BAsYPIrB/view) | Metadata saja |
| 115 | [Effects of embedded questions in recorded lectures.pdf](https://drive.google.com/file/d/1NZHZpz66ywy7CTDekrwCd_S_HA3_io2s/view) | Metadata saja |
| 116 | [Sequence matters! Retrieval practice before generative learning is more effective than the reverse order.pdf](https://drive.google.com/file/d/1U1Zo_NiNfC6yJmF3F6YwtM0bZhlMHk2W/view) | Metadata saja |
| 117 | [The effects of using an auto‐subtitle system in educational videos to facilitate learning for secondary school students_ learning comprehension, cognitive load, and satisfaction.pdf](https://drive.google.com/file/d/17jX0o1E62Jmaa9OMrVTUUwvDkOh9WeTr/view) | Metadata saja |
| 118 | [Review on Neural Question Generation for Education Purposes.pdf](https://drive.google.com/file/d/1ZXmtcXnTqIOHUDNepOuP7HPRA6Xv3Lpl/view) | Metadata saja |
| 119 | [SUMMAC_ Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.pdf](https://drive.google.com/file/d/19_tuf0qEoNcpBe_G2hfIPb-SoH8EHR2U/view) | Metadata saja |
| 120 | [ALIGNSCORE_ Evaluating Factual Consistency with A Unified Alignment Function.pdf](https://drive.google.com/file/d/1508xtukBbG5mNtjOCgglh--NmAexuV4D/view) | Metadata saja |
| 121 | [QAFactEval_ Improved QA-Based  Factual Consistency Evaluation for Summarization.pdf](https://drive.google.com/file/d/1HmMJ5JxcAgcFbzmU8RiWttcLWRDZhz5k/view) | Metadata saja |
| 122 | [Proposal_Final_LeveLearn_Bab_I-III.docx](https://drive.google.com/file/d/1bkhv2CaiYAMKbPaAYyLLxvEKc-8tNJUV/view) | Metadata saja |
| 123 | [Does adding video and subtitles to an audio lesson facilitate its comprehension_.pdf](https://drive.google.com/file/d/1cQTOTefTHk1HUiguNztMYBAlQvQ9HsFZ/view) | Metadata saja |
| 124 | [Automating True-False Multiple-Choice Question Generation and Evaluation with Retrieval-based Accuracy Differential.pdf](https://drive.google.com/file/d/1S-EsbyPx4VQbwFZvljlzsUpFGKvpLTi4/view) | Metadata saja |
| 125 | [Unsupervised Lexical Simplification with Context Augmentation.pdf](https://drive.google.com/file/d/1jv4l8Eg2H0DCs8DyyY3uZOIZTfH6X-4a/view) | Metadata saja |
| 126 | [Learning from interactive video_ the influence of self- explanations, navigation, and cognitive load.pdf](https://drive.google.com/file/d/13uHJfS-YfASddpTnJk0SeNP9kzpni3JV/view) | Metadata saja |
| 127 | [FENICE_ Factuality Evaluation of Summarization Based on Natural Language Inference and Claim Extraction.pdf](https://drive.google.com/file/d/1xMaXtHfsqYnzCcjqDGrxa5dHwoEGX_gO/view) | Metadata saja |
| 128 | [A Feasibility Study of Answer-Agnostic Question Generation for  Education.pdf](https://drive.google.com/file/d/12Rsb09lATI4gwwN5NDojXd7UMhzPJn8A/view) | Metadata saja |
| 129 | [Short, Long, and Segmented Learning Videos_ From YouTube Practice to Enhanced Video Players.pdf](https://drive.google.com/file/d/1FKAsDNPpqA1XJsmh5nXW7F5rf-sQJJRv/view) | Metadata saja |
| 130 | [Planning First, Question Second_ An LLM-Guided Method for  Controllable Question Generation.pdf](https://drive.google.com/file/d/1ATfcgS_8C3L_yAT_P6zJkO_F24AisPSH/view) | Metadata saja |
| 131 | [ALEXSIS+_ Improving Substitute Generation and Selection for Lexical  Simplification with Information Retrieval.pdf](https://drive.google.com/file/d/1V2YKx73Xm6WJaZvoz7MAXTTqK4Odi6TM/view) | Metadata saja |
| 132 | [Question Generation for Adaptive Education.pdf](https://drive.google.com/file/d/1zfsaBPt60U1d9TcjiTFjyWY7wksb7eoe/view) | Metadata saja |
| 133 | [On Faithfulness and Factuality in Abstractive Summarization.pdf](https://drive.google.com/file/d/1mKX_k20YP3tG5ipVypMFe_Ty7d3l3RVz/view) | Metadata saja |
| 134 | [Jurnal Practical Application of an e-Learning Support System.pdf](https://drive.google.com/file/d/1jDxp9GiF5FUSDr41btGHC4YJw-dbBNgK/view) | Metadata saja |
| 135 | [Jurnal Eyes Wide Open (1).pdf](https://drive.google.com/file/d/17SuHAufeMHbl2DctCSlPBQncY24Xhoe5/view) | Metadata saja |
| 136 | [1.pdf](https://drive.google.com/file/d/1Sj4p2h24n9MRvVTvSTMjQJOapEl8Bmcv/view) | Metadata saja |
| 137 | [Jurnal Grounded-VideoLLM Sharpening.pdf](https://drive.google.com/file/d/1fsCGekCSgt70In8YvHr8wDzxwYEicWjV/view) | Metadata saja |
| 138 | [Jurnal Watch, Remember, Reason.pdf](https://drive.google.com/file/d/1b20KqA2SHLWK2uc_xtmQy-LLMFV4Xg4n/view) | Metadata saja |
| 139 | [paper.pdf](https://drive.google.com/file/d/1oannAyKeBbfYDxccsc8u-yWcaVDGOa3-/view) | Metadata saja |
| 140 | [Jurnal Video Understanding with Large Language Models (1).pdf](https://drive.google.com/file/d/1hG-_FArsmL_S9P1JlGit6kcbA1gaQEhR/view) | Metadata saja |
| 141 | [kim14video.pdf](https://drive.google.com/file/d/18UJHbIYqYv-TbGil58D90SqIS008oDJl/view) | Metadata saja |
| 142 | [Jurnal Can I Trust Your Answer (2).pdf](https://drive.google.com/file/d/1Mo22Mr-bPAgNIamUpvPoHKqrub0EtQ13/view) | Metadata saja |
| 143 | [Jurnal The implementation of the cognitive theory of multimedia.pdf](https://drive.google.com/file/d/1eDTAmjPodX7Wjeu2mHsS85XNt_M5yOrF/view) | Metadata saja |
| 144 | [Jurnal ViSIL Unified Evaluation of Information Loss in Multimodal Video Captioning (1).pdf](https://drive.google.com/file/d/1gLDXqMcdS9FZDKe5qSl9hcKQIfczi3GN/view) | Metadata saja |
| 145 | [Jurnal Immediate Versus Delayed Low‑Stakes Questioning.pdf](https://drive.google.com/file/d/10x2o5iTmW0ZBULa2HMw4jIs8r1hy7faY/view) | Metadata saja |
| 146 | [Jurnal Implications of Designing Instructional Video Using.pdf](https://drive.google.com/file/d/1W5bb4fUslSdJTnF7mN2yRX3-fmRvYdza/view) | Metadata saja |
| 147 | [Jurnal Enhancing the Learning Experience.pdf](https://drive.google.com/file/d/1zSwRXgwo2Cdj3Ppk3L6I4zjgq-AkKIJn/view) | Metadata saja |
| 148 | [Jurnal Overview of the NLPCC 2026 Shared Task (2).pdf](https://drive.google.com/file/d/1W0lM4Sc7Eg8maWITjkLyNsctG1i4M9Mb/view) | Metadata saja |
| 149 | [Jurnal Reflective Experiential Learning.pdf](https://drive.google.com/file/d/189zALTjSSnXUBj0k_wuT57mqEa0Wj48d/view) | Metadata saja |
| 150 | [9_ways_to_reduce_CL.pdf](https://drive.google.com/file/d/1oY1ZWy3RIXQ4pcxnQe1NC-XeAi3GrX31/view) | Metadata saja |
| 151 | [Jurnal Automatically Generated Summaries of Video Lectures (1).pdf](https://drive.google.com/file/d/1_lzcO3b082golmzsdtlXyhRVbdk0pUGs/view) | Metadata saja |
| 152 | [Visual Transcripts Lecture Notes from Blackboard-Style Lecture Videos.pdf](https://drive.google.com/file/d/1F89VuxFHAtCb20BzvkXXHpSPrZKlz88f/view) | Metadata saja |
| 153 | [Grounded-VideoLLM_ Sharpening Fine-grained Temporal Grounding in  Video Large Language Models.pdf](https://drive.google.com/file/d/1eaT5A_yM8BS3kPFXAk1NREkhMj-GFA-Z/view) | Metadata saja |
| 154 | [Context Selection and Rewriting for Video-based Educational Question Generation(1).pdf](https://drive.google.com/file/d/1XDcYhVrmhRBd-DrqFHwR47dGooUQwPCm/view) | Metadata saja |
| 155 | [NExT-QA Next Phase of Question-Answering to Explaining Temporal Actions.pdf](https://drive.google.com/file/d/1ByFDGXZKcy060GZkKzt4JZ1A6iMQbd38/view) | Metadata saja |
| 156 | [Human-View Video Understanding with MLLMs (9).pdf](https://drive.google.com/file/d/1Baz93dWocWG23-iIAijoqO021YsNDb0d/view) | Metadata saja |
| 157 | [A roadmap for medical large language models (8).pdf](https://drive.google.com/file/d/1sYKXsylqFeu9Sxc-aXtobUHZo6-Ynhij/view) | Metadata saja |
| 158 | [V-Rex (7).pdf](https://drive.google.com/file/d/1RXkUMoER7odOwKkRtR6d5d3UMibLweEI/view) | Metadata saja |
| 159 | [LLaVA-Plus Learning to Use Tools for Creating Multimodal Agents (4).pdf](https://drive.google.com/file/d/1W-ldnqkjlOIhxFcm6cZVmCBmZuqAmjE0/view) | Metadata saja |
| 160 | [Overview of the NLPCC 2026 Shared Task 1 Difficulty-Aware Multilingual and Multimodal Medical Instructional Video Understanding Evaluation (3).pdf](https://drive.google.com/file/d/1bvO_4FV5xTMpUxNQxKXCmTiBQ3cB_4gW/view) | Metadata saja |
| 161 | [Digital twin improved via visual question answering for vision-language interactive mode in human–machine collaboration. (2020) (2)..pdf](https://drive.google.com/file/d/1Zvbe0gOJ_oIuof6ddwupAJFhxyQONquW/view) | Metadata saja |
| 162 | [Eyes Wide Open (6).pdf](https://drive.google.com/file/d/19oxPd0FYUXqu6iWyWjHtVgEbRWxb5TBX/view) | Metadata saja |
| 163 | [Mathematical Framework for Explainable Vehicle Systems Integrating Graph-Theoretic Road Geometry and Constrained Optimization (10).pdf](https://drive.google.com/file/d/1B5PJX6jMPq5GtUoIyMbVRO_lYY8oXDkV/view) | Metadata saja |
| 164 | [Unified Evaluation of Information Loss in Multimodal Video Captioning. (2026) (5).pdf](https://drive.google.com/file/d/164RSy3GQCBRatkIuXZPKEUrELisxP22A/view) | Metadata saja |
| 165 | [Video Understanding with Large Language Models (1).pdf](https://drive.google.com/file/d/1n3q-ku3lsx2XMbfyWoZrgrxWoLkoOXhB/view) | Metadata saja |

