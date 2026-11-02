# Konteks fine-tuning Qwen3.5-9B untuk LeveLearn

Tanggal: 2 November 2026, Asia/Jakarta.  
Penerima: Claude atau asisten pemrograman yang akan membantu implementasi.  
Tujuan dokumen: menjelaskan tujuan, keputusan, percobaan yang sudah dilakukan, data yang perlu disiapkan, dan pekerjaan selanjutnya. Dokumen ini bukan laporan bahwa fine-tuning multimodal telah selesai.

## 1. Ringkasan tugas untuk Claude

Saya sedang mengembangkan fitur LeveLearn dalam penelitian Tugas Akhir. Saya ingin mengadaptasi checkpoint **Qwen/Qwen3.5-9B** agar dapat membangkitkan **soal pilihan ganda, kunci jawaban, dan pembahasan jawaban** dari segmen video pembelajaran Logika Proposisi.

Masukan utama yang direncanakan adalah **frame video terurut + transkrip ucapan pada segmen + instruksi tugas**. Bukan hanya pertanyaan teks, bukan gambar tanpa konteks, dan bukan melatih model dari nol.

Saya sudah menjalankan percobaan kecil berbasis teks dengan kuantisasi 4-bit dan LoRA pada Google Colab Tesla T4. Percobaan itu menunjukkan model dapat dimuat dan parameter adapter dapat diperbarui. Percobaan tersebut **belum membuktikan fine-tuning multimodal berjalan atau meningkatkan kualitas**.

Sekarang saya ingin menyiapkan pilot satu segmen video, memeriksa inferensi multimodal model dasar, lalu menyiapkan data teranotasi dan mencoba fine-tuning yang sesuai dengan tugas sebenarnya. Saya juga mempertimbangkan pengerjaan melalui VS Code di komputer lokal. Spesifikasi perangkat lokal belum diberikan.

Tolong bantu secara bertahap, dengan kode yang dapat dijalankan dan penjelasan bahasa Indonesia sederhana. Pertahankan keputusan yang sudah jelas, dan bedakan hasil nyata, rencana penelitian, serta usulan teknis.

## 2. Model yang ditetapkan

| Aspek | Keputusan atau penjelasan |
|---|---|
| ID checkpoint | `Qwen/Qwen3.5-9B` |
| Sumber resmi | https://huggingface.co/Qwen/Qwen3.5-9B |
| Jenis | Model bahasa dengan vision encoder; menerima informasi visual dan teks |
| Tahap checkpoint | Post-trained; bukan `Qwen3.5-9B-Base` |
| Ukuran | Kelas 9B, sekitar sembilan miliar parameter; total parameter yang dihitung implementasi dapat berbeda dari label ukuran |
| Lisensi pada halaman resmi | Apache-2.0; tetap periksa ketentuan data dan dependensi secara terpisah |
| Adaptasi | Supervised fine-tuning menggunakan pasangan masukan dan target yang telah diperiksa |
| Strategi hemat memori awal | Bobot utama 4-bit dan adapter LoRA; konfigurasi akhir mengikuti hasil pilot |

Halaman resmi diperiksa pada 2 November 2026 untuk identitas checkpoint, status post-trained, tipe model, dan lisensi. Jangan menerapkan klaim arsitektur seluruh keluarga Qwen secara otomatis pada varian 9B. Verifikasi konfigurasi checkpoint dan dokumentasi saat membuat kode.

Istilah penting:

- **VLM**: vision-language model, model yang mengolah visual dan bahasa.
- **Fine-tuning**: menyesuaikan model yang sudah terlatih menggunakan data tugas tertentu.
- **LoRA**: melatih parameter tambahan berukuran kecil, sementara bobot utama dibekukan.
- **Kuantisasi 4-bit**: menyimpan bobot dengan presisi lebih rendah untuk menghemat memori; bukan berarti seluruh proses hanya membutuhkan empat bit per parameter.
- **Inferensi**: menggunakan model untuk menghasilkan keluaran, tanpa memperbarui parameter.

Membekukan vision encoder tidak otomatis menjadikan pelatihan teks saja: masukan tetap multimodal bila frame benar-benar diproses dan memengaruhi prediksi. Namun, strategi pembekuan dan modul yang dilatih harus dilaporkan dengan jelas.

## 3. Tujuan dan ruang lingkup LeveLearn

Judul yang tercatat dalam keputusan proyek 31 Oktober:

> Pembangkitan Soal Pilihan Ganda dan Pembahasan Jawaban dari Video Pembelajaran Logika Proposisi Berbasis Vision-Language Model pada LeveLearn.

LeveLearn sudah memiliki fondasi aplikasi. Penelitian berfokus pada pengembangan fitur pengolahan video dan latihan, bukan membangun LMS dari nol.

Materi yang direncanakan mencakup formula dan interpretasi/tabel kebenaran, ekuivalensi logis dan semantic tableaux, deduksi Gentzen, deduksi Hilbert, resolusi proposisi, serta Binary Decision Diagram (BDD). Predikat dan kuantor bukan korpus utama.

Tujuan teknis:

1. Menghasilkan soal yang relevan dengan segmen, empat opsi, dan tepat satu jawaban benar.
2. Menghasilkan pembahasan yang benar secara logika, jelas, dan sesuai sumber.
3. Menguji kontribusi masukan visual dan fine-tuning terhadap kelayakan soal, ketepatan kunci, serta ketepatan pembahasan dan notasi.

Jangan menambahkan summarization sebagai tugas keluaran utama terpisah. Waktu awal–akhir adalah metadata segmen, bukan tugas memprediksi interval temporal grounding.

## 4. Alur keluaran: QG dan AE terpisah

Keputusan penelitian mencatat dua tahap:

1. **Question Generation (QG)**: satu pemanggilan per segmen untuk menghasilkan tiga soal pilihan ganda, masing-masing empat opsi dan satu kunci.
2. **Answer Explanation (AE)**: satu pemanggilan per soal, menggunakan konteks segmen, soal, opsi, dan kunci untuk menghasilkan pembahasan.

Tiga soal valid berarti satu pemanggilan QG dan tiga pemanggilan AE, yaitu empat pemanggilan sebelum retry. Ini tidak mensyaratkan dua model atau dua adapter: satu checkpoint/adapter dapat dipakai dengan instruksi tugas berbeda, jika itulah implementasi yang dipilih.

Pembahasan wajib memuat:

- Alasan jawaban benar dan langkah yang diperlukan.
- Notasi, nilai kebenaran, dan aturan inferensi yang tepat.
- Alasan singkat setiap pengecoh salah.
- ID segmen dan waktu awal–akhir yang diambil dari metadata.
- Bahasa sederhana; istilah teknis dijelaskan saat pertama digunakan.

Pengayaan di luar sumber harus diberi label dan diperiksa pengajar. Jangan mengklaim suatu hal dijelaskan dalam video bila tidak ada buktinya. Batas panjang pembahasan final belum ditetapkan; jangan memotong penalaran penting demi angka kata arbitrer.

Alur publikasi: keluaran mentah → validasi → draf → pemeriksaan pengajar → publikasi. Evaluasi model memakai keluaran mentah; konten untuk mahasiswa memakai versi yang sudah diperiksa.

## 5. Bukti percobaan awal yang sudah ada

Notebook: https://colab.research.google.com/drive/1L0HNH-e7SZf0dmqLl-41rkdtZoDN0Jbc?usp=sharing  
Nama: `Latihan_Qwen_LeveLearn.ipynb`.

Informasi di bawah berasal dari kode dan keluaran tersimpan yang dibaca dalam percakapan sebelumnya. Ini bukan hasil menjalankan ulang notebook pada 2 November.

| Komponen | Konfigurasi atau hasil tersimpan |
|---|---|
| GPU | Tesla T4, memori dilaporkan 14,6 GiB |
| Model lokal Colab | `/content/drive/MyDrive/LeveLearn/models/Qwen3.5-9B` |
| Berkas bobot | Empat berkas `.safetensors` ditemukan |
| Kelas model | `Qwen3_5ForConditionalGeneration` |
| Processor | `AutoProcessor` |
| Kuantisasi | NF4 4-bit, double quantization aktif |
| Tipe komputasi | `torch.float16` |
| Perangkat | `device_map={"": 0}` |
| Attention implementation | `sdpa` |
| Memori setelah dimuat | Sekitar 7,35 GiB memori teralokasi, bukan puncak kebutuhan pelatihan |
| Data | Enam pasangan pertanyaan–jawaban teks tentang konjungsi, disjungsi, negasi |
| LoRA | `r=4`, `lora_alpha=8`, `lora_dropout=0.0`, `bias="none"` |
| Sasaran LoRA | Modul yang mengandung `.self_attn.` dan berakhiran `.q_proj`/`.v_proj` |
| Parameter dapat dilatih | 557.056; sekitar 0,0059% dari total yang dilaporkan |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Langkah | 12 |
| Learning rate | `1e-4` |
| Optimizer | `adamw_torch` |
| Precision | `fp16=True`, `bf16=False` |
| Hemat aktivasi | Gradient checkpointing, `use_reentrant=False` |
| Seed | 42 |
| Simpan checkpoint | Setiap enam langkah; maksimal dua checkpoint |
| Output tersimpan | `/content/drive/MyDrive/LeveLearn/latihan_dummy/20261031_133615` |

Versi yang dilaporkan notebook: Transformers 5.22.1, bitsandbytes 0.50.2; keluaran instalasi juga mencantumkan PEFT 0.21.2 dan accelerate 1.15.0. Ini catatan historis lingkungan Colab, bukan jaminan kompatibel pada semua komputer atau rekomendasi memasang tanpa pemeriksaan.

Data latihan dibuat menggunakan chat template dengan `enable_thinking=False`. Bagian prompt diberi label `-100` agar tidak dihitung dalam loss; target jawaban diikuti EOS. Panjang contoh dibatasi 256 token. `default_data_collator` dipakai dengan batch size 1; saat batch diperbesar, perlu padding dan masking yang benar.

### Hasil tersimpan dan maknanya

- Semua nilai LoRA B yang diperiksa finite; nilai absolut terbesar sekitar `0.0004182866250630468`.
- Loss pada enam data latihan tanpa adapter: `2.218250`.
- Loss pada data yang sama dengan adapter: `2.140466`.
- Selisih: `-0.077784`.
- Jawaban generasi sebelum dan sesudah latihan pada tiga soal uji tetap sama.
- Generasi menggunakan `do_sample=False`, `max_new_tokens=150`, dan beberapa jawaban terpotong. Batas token perlu diperiksa, bukan langsung menyimpulkan model tidak dapat menyelesaikan penjelasan.
- Ada peringatan fallback implementasi PyTorch karena kernel opsional `causal_conv1d` dan `flash-linear-attention` tidak tersedia. Periksa kompatibilitas sebelum memasangnya.

Kesimpulan yang sah: pembaruan adapter terjadi dan kecocokan pada data latihan sedikit meningkat. Belum ada bukti peningkatan generalisasi, kualitas soal, atau pemahaman video. Loss tersebut bukan persentase akurasi.

Sasaran LoRA hanya sebagian modul yang dipilih berdasarkan nama; jangan mengklaim semua blok attention atau vision encoder telah dilatih. Verifikasi daftar modul, gradien, dan parameter yang benar-benar menerima pembaruan.

## 6. Pilot video yang dipilih pengguna

Folder Drive: https://drive.google.com/drive/folders/1Nd2pPcZQk3S9frXUwBcI3TleGjUnqaBS?usp=sharing

File yang ditemukan:

- Nama: `Propositional Logic_ Binary Decision Diagram 1080P.mp4`.
- Drive file ID: `19Vtm2DHc2t7L-9Nmr_KEqIg0hxxs04Pp`.
- Durasi video penuh sekitar 27 menit menurut pengguna; belum diverifikasi melalui metadata media.
- Segmen yang ditetapkan pengguna: **04:10–06:27**.
- Awal absolut: 250 detik; akhir: 387 detik; durasi: **137 detik / 2 menit 17 detik**.

Menurut pengguna, bagian ini membahas menyederhanakan tabel panjang menjadi bentuk lebih sederhana dalam pembahasan BDD. Screenshot menunjukkan tabel kebenaran `p ∨ (q ∧ r)` diringkas dari delapan menjadi empat baris. Tanda `*` mewakili nilai benar atau salah yang tidak memengaruhi hasil pada kondisi tersebut. Gambar itu sendiri belum merupakan diagram BDD; jangan menyimpulkan semua langkah konstruksi BDD ditampilkan tanpa membaca video/transkrip.

Status pelaksanaan: kode pemotongan sudah diberikan, tetapi belum ada konfirmasi pengguna bahwa pemotongan, ekstraksi frame, ASR, atau inferensi multimodal berhasil dijalankan. Jangan menganggap file hasil sudah tersedia.

Kode terakhir mengasumsikan Drive sudah di-mount, mencari nama video di `/content/drive/MyDrive`, memotong dengan FFmpeg memakai `-ss 00:04:10` dan `-t 137`, lalu menyimpan ke `MyDrive/LeveLearn/pilot/segmen_0410_0627_<waktu>.mp4`. Jangan mengulang autentikasi/mount jika sesi masih aktif dan jalur dapat diakses.

Video pilot ini dipakai untuk menguji alur. Menurut catatan keputusan 31 Oktober, korpus penelitian utama direncanakan berupa rekaman pembelajaran IT Del dengan izin; video eksternal yang tersedia sekarang tidak otomatis menggantikan keputusan sumber utama. Bahasa audio pilot belum dikonfirmasi. Jangan memaksa ASR bahasa Indonesia hanya karena keluaran akhir diminta bahasa Indonesia.

## 7. Bentuk data yang perlu dibangun

### 7.1 Masukan per segmen

1. ID video, ID segmen, kelompok sumber, waktu awal–akhir, dan asal file.
2. Frame terurut dengan timestamp, cukup jelas untuk membaca formula, tabel, dan langkah penalaran.
3. Transkrip ucapan yang benar-benar sesuai interval, beserta timestamp.
4. Instruksi tugas QG atau AE.

Audio diubah menjadi teks oleh komponen ASR terpisah dalam alur ini. Jangan menganggap Qwen sudah mendengar audio hanya karena menerima gambar. File MP4 dapat menjadi sumber prapemrosesan; pendekatan yang sedang dibahas mengirim frame dan teks ke model. Dukungan jalur input video native dapat dieksplorasi setelah diverifikasi, tetapi bukan syarat pilot ini.

Jumlah frame, resolusi, metode sampling, dan model ASR belum dibekukan. Mulai dengan jumlah kecil yang terukur sesuai kapasitas GPU, periksa keterbacaan, lalu tentukan konfigurasi melalui pilot. Hindari frame duplikat yang boros dan kehilangan slide penting. Timestamp frame harus dibedakan antara relatif segmen dan absolut video.

OCR adalah teks hasil pengenalan tulisan dari frame. Siapkan terutama sebagai kondisi pembanding; jangan menyamakan OCR dengan transkrip ucapan.

### 7.2 Target data latihan

Untuk QG, target berisi tiga soal yang diperiksa, empat opsi tiap soal, dan satu kunci. Untuk AE, target berisi pembahasan yang diperiksa untuk satu soal, termasuk alasan kunci dan pengecoh.

Target disusun/diperiksa oleh manusia yang memahami materi. Draf AI boleh membantu jika kebijakan penelitian mengizinkan, tetapi jangan menjadikan keluaran AI yang belum diperiksa sebagai ground truth.

Satu segmen bisa menghasilkan satu contoh QG dan tiga contoh AE. Empat contoh tersebut tetap berasal dari satu sumber yang sama, bukan empat sumber independen.

Contoh berikut adalah **skema konseptual**, bukan format final siap dimasukkan ke Trainer:

```json
{
  "example_id": "pilot_bdd_001_qg",
  "task": "qg",
  "source_group_id": "pilot_bdd_source_01",
  "video_id": "pilot_bdd_01",
  "segment_id": "pilot_bdd_01_s001",
  "start_seconds": 250,
  "end_seconds": 387,
  "frames": [
    {"path": "frames/frame_001.jpg", "timestamp_seconds_absolute": 255}
  ],
  "transcript": "<transkrip asli segmen yang telah diperiksa>",
  "instruction": "Buat 3 soal pilihan ganda berdasarkan konteks; setiap soal memiliki 4 opsi dan tepat 1 kunci.",
  "target": {"questions": "<daftar 3 soal tervalidasi>"},
  "annotation_status": "draft"
}
```

Path dan timestamp frame pada contoh hanyalah ilustrasi, bukan bukti frame tersebut sudah diekstrak. Untuk AE tambahkan soal, opsi, dan kunci pada masukan; target berupa pembahasan terstruktur. Bentuk JSON keluaran final perlu ditetapkan dan divalidasi.

### 7.3 Perbedaan latihan, validasi, dan uji

| Pembagian | Target tersedia bagi model? | Kegunaan |
|---|---|---|
| Train | Ya, sebagai label pembelajaran | Memperbarui adapter |
| Validation | Sebagai label evaluasi loss bila digunakan; bukan bagian prompt generasi | Memilih konfigurasi/checkpoint |
| Test | Tidak diberikan dalam prompt; disimpan untuk penilai | Menilai kemampuan pada sumber terpisah |

Pisahkan berdasarkan kelompok video/sumber/slide terkait sebelum membentuk banyak contoh. Semua frame, varian prompt, QG dan AE dari sumber yang sama harus berada pada pembagian yang sama. Hindari membagi segmen bertetangga secara acak ke train dan test. Data yang dipakai menyetel pilot jangan digunakan kembali untuk klaim uji akhir yang independen.

Target jumlah dalam catatan penelitian adalah sedikitnya 40 video atau 300 segmen dari sedikitnya 15 sumber independen. Ini target perencanaan, bukan jumlah data tersedia. Rasio split dan data aktual belum ditetapkan.

## 8. Persyaratan implementasi fine-tuning multimodal

Ini kebutuhan implementasi yang perlu diwujudkan, bukan klaim kode sudah tersedia:

1. Gunakan processor model untuk memuat dan mengubah gambar serta teks menjadi tensor yang benar.
2. Gunakan chat template yang konsisten antara training dan inferensi.
3. Pastikan batch berisi informasi visual aktual, bukan sekadar nama file atau token gambar tanpa data gambar.
4. Terapkan masking label prompt, padding, dan posisi visual sesuai dokumentasi processor/model. Jangan menebak ID token visual.
5. Hitung loss pada target assistant yang relevan; periksa ada token target valid dan gradien finite.
6. Bekukan bobot utama sesuai strategi LoRA; verifikasi nama modul sasaran pada checkpoint ini.
7. Pilih batch size, resolusi/jumlah frame, panjang token, rank LoRA, dan gradient accumulation melalui pilot memori. Nilai dummy sebelumnya bukan konfigurasi optimal final.
8. Simpan adapter, processor/tokenizer yang diperlukan, konfigurasi, ID/revisi checkpoint awal, versi paket, seed, dan metadata split.
9. Uji memuat ulang checkpoint awal + adapter dari sesi baru dan memeriksa inferensi dengan masukan yang sama.
10. Catat peak VRAM, waktu pelatihan, waktu generasi, kegagalan/OOM, jumlah contoh, dan jumlah token.

Simpan keluaran mentah serta keluaran hasil parsing secara terpisah. Parser gagal bukan alasan menghapus contoh tanpa pencatatan. Retry harus mempunyai aturan tetap.

## 9. Eksekusi lokal melalui VS Code

Pengguna bertanya bagaimana melakukan latihan di komputer lokal. Hal yang belum diketahui: OS, GPU, VRAM, RAM, ruang disk, driver, dan Python. Jangan menganggap GPU lokal sama dengan T4 atau menjanjikan model muat.

VS Code adalah editor/antarmuka. Komputasi bisa berada di komputer lokal atau server GPU melalui akses remote yang diizinkan. Catatan penelitian menyebut rencana akses B200 IT Del, tetapi alokasi, akses aktual dan jam penggunaan belum terverifikasi. Jangan menganggap sumber daya itu sudah tersedia di laptop pengguna.

Persiapan setelah spesifikasi diketahui:

- Python environment terpisah dan interpreter/kernel VS Code yang sesuai.
- Instalasi PyTorch sesuai GPU/OS, kemudian Transformers, PEFT, accelerate, bitsandbytes, dan dependensi prapemrosesan yang kompatibel.
- FFmpeg untuk pemotongan serta ekstraksi audio/frame.
- Folder model lengkap, data, anotasi dan output lokal.
- Hapus sel khusus Colab seperti `drive.mount()`/`auth.authenticate_user()` pada jalur lokal.
- Ganti `/content/drive/MyDrive/...` dengan jalur lokal; gunakan `pathlib.Path`.
- Bedakan perintah terminal `python -m pip install ...` dan magic notebook `%pip install ...`.
- Simpan adapter dan log secara permanen; jangan bergantung pada `/content` untuk jalur lokal.

Contoh lokasi `D:/LeveLearn/models/Qwen3.5-9B` sebelumnya hanya ilustrasi, bukan jalur komputer pengguna yang sudah diketahui.

Jangan menetapkan angka VRAM minimum pasti tanpa konfigurasi dan pengukuran. Memori memuat bobot saja bukan memori training. Aktivasi visual, panjang konteks, optimizer dan buffer ikut berpengaruh.

## 10. Evaluasi yang dibutuhkan

Bandingkan checkpoint awal dan hasil fine-tuning dengan segmen uji, prompt, serta konfigurasi generasi yang sama. Untuk kontribusi visual, bedakan kondisi transkrip saja, transkrip + OCR tanpa frame, dan transkrip + frame; kendalikan variabel lain. Pembanding komersial adalah bagian rancangan lebih luas, bukan prasyarat memulai pilot lokal; ID API historis harus diverifikasi ulang sebelum dipakai.

Penilaian mencakup:

- Format valid: jumlah soal/opsi dan satu kunci.
- Kelayakan: relevan, jelas, tidak ambigu, dapat dijawab dari sumber.
- Kebenaran kunci dan pengecoh.
- Ketepatan pembahasan dan notasi: operator, negasi, tanda kurung, nilai kebenaran dan aturan inferensi.
- Dukungan isi oleh segmen; kesalahan/hallucination dicatat.
- Kelengkapan alasan setiap opsi dan kejelasan bahasa.
- Kecepatan dan kebutuhan memori.

Jika memakai Symbol Error Rate (SER), tetapkan satuan kesalahan dan penyebut secara konsisten sebelum penilaian. Jangan menjadikan penurunan loss latihan satu-satunya bukti keberhasilan atau mengklaim fine-tuning pasti unggul.

Penilaian pengajar atas keluaran model dipisahkan dari uji keterpahaman mahasiswa. Dokumen ini fokus implementasi fine-tuning; detail protokol mahasiswa harus mengikuti proposal terbaru.

## 11. Urutan bantuan yang saya butuhkan dari Claude

1. Konfirmasi pemahaman tujuan dalam beberapa kalimat; jangan bertanya ulang model yang sudah dipilih.
2. Jika akan memakai lokal, minta spesifikasi perangkat yang benar-benar belum ada. Jika tetap Colab, gunakan konfigurasi yang telah terbukti sebagai titik awal.
3. Periksa apakah segmen pilot 04:10–06:27 sudah berhasil dibuat; bila belum, bantu pemotongannya tanpa menimpa sumber.
4. Ekstrak frame dan transkrip; tampilkan sampel untuk verifikasi isi, bahasa audio, notasi dan timestamp.
5. Jalankan satu inferensi multimodal checkpoint dasar. Buktikan frame termuat dan keluaran lengkap sebelum melatih.
6. Siapkan skema anotasi QG dan AE, contoh acuan yang diperiksa, serta manifest split berbasis sumber.
7. Buat training script/notebook yang benar-benar menerima visual dan teks, dengan uji satu batch forward/backward sebelum training lebih panjang.
8. Jalankan pilot fine-tuning sesuai perangkat; simpan dan muat ulang adapter.
9. Bandingkan model dasar dan adapter pada data terpisah; laporkan hasil positif maupun negatif.

Berikan kode bertahap dengan tujuan, masukan, hasil yang diharapkan dan cara memeriksa keberhasilannya. Jangan menjalankan pipeline penuh secara membabi buta saat sumber/path belum tersedia.

## 12. Batas pengetahuan dan hal yang belum selesai

- Spesifikasi komputer lokal belum ada.
- Belum ada konfirmasi potongan video berhasil dibuat.
- Belum ada hasil ASR/frame pilot atau inferensi multimodal dalam percakapan ini.
- Belum ada dataset fine-tuning multimodal yang diperiksa dan siap pakai.
- Belum ada split final, hyperparameter multimodal final, atau hasil evaluasi uji independen.
- Belum ada bukti model hasil latihan lebih baik dari checkpoint awal.
- Ukuran korpus, izin tertulis, identitas penilai dan fasilitas GPU aktual tetap perlu didokumentasikan.
- Catatan proyek yang dibaca untuk dokumen ini bertanggal 31 Oktober, sedangkan ringkasan percakapan menunjukkan ada berkas Bab III lebih baru pada 1 November. Isi berkas lebih baru itu tidak diperiksa di pekerjaan ini. Untuk klaim metodologi final atau perubahan proposal, baca versi terbaru; jangan menganggap catatan 31 Oktober mengalahkan keputusan eksplisit yang lebih baru.

## 13. Sumber dan keterlacakan

1. Percakapan pengguna tentang notebook, bentuk data VLM, pilot video BDD, mounting Drive dan rencana VS Code; sampai 2 November 2026.
2. Notebook `Latihan_Qwen_LeveLearn.ipynb` pada tautan di bagian 5; rincian berdasarkan pembacaan sebelumnya, bukan eksekusi ulang.
3. Daftar file folder Drive yang pernah dibaca; nama dan ID video tercantum pada bagian 6. Isi seluruh video belum diperiksa.
4. Screenshot tabel `p ∨ (q ∧ r)` yang diberikan pengguna; mendukung penjelasan isi visual, bukan bukti keseluruhan narasi video.
5. `Konteks_Keputusan_Bab_I_II_dan_Revisi_Bab_III_31_Oktober_2026.md`, sumber proyek yang dibaca untuk menyelaraskan keputusan penelitian.
6. Model card resmi: https://huggingface.co/Qwen/Qwen3.5-9B, diperiksa 2 November 2026 untuk identitas model dasar pada bagian 2.

**Pesan penutup untuk Claude:** Fokus membantu saya berpindah dari percobaan LoRA teks kecil menuju pilot dan fine-tuning multimodal yang benar untuk QG–AE LeveLearn. Pertahankan model utama, bedakan yang sudah berhasil dari yang masih rencana, dan jangan mengarang hasil eksperimen atau spesifikasi perangkat.
