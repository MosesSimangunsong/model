A. Izin akses (agar skill berisi fakta terverifikasi, bukan tebakan)
A1. Bolehkah saya membaca file konfigurasi model lokal di Model/Qwen3.5-9B/? Yang dibaca hanya file teks kecil: config.json, chat_template.jinja, preprocessor_config.json, video_preprocessor_config.json, dan model.safetensors.index.json (daftar nama parameter — dari sini saya bisa memastikan nama modul target LoRA tanpa memuat bobot 18 GB).

Jawaban: Boleh. Baca konfigurasi yang relevan untuk memverifikasi struktur model dan calon modul target LoRA, tanpa mengubah file model. Tetap verifikasi nama modul pada implementasi model sebelum memasang LoRA.

A2. Bolehkah saya mengkloning 2U1/Qwen-VL-Series-Finetune ke folder sementara (scratchpad, di luar repo Anda) untuk membaca cara mereka menangani Qwen3.5 (collator, masking label, modul yang dibekukan)? Tidak ada yang diinstal atau dijalankan.

Jawaban: Boleh, untuk dibaca sebagai referensi di folder sementara di luar repo saya. Jangan otomatis menginstal atau menjalankannya. Verifikasi dukungan Qwen3.5 dan kesesuaian implementasinya dengan checkpoint yang digunakan.
B. Ruang lingkup skill
B1. Skill mencakup apa saja?
- (a) Hanya training (format data → collator → LoRA → uji 1 batch → training → simpan/muat ulang adapter).
- (b) Pipeline penuh: prapemrosesan (potong video, frame, ASR) + format data + training + inferensi + evaluasi.
- (c) Dipecah menjadi beberapa skill kecil (mis. levelearn-preprocess, qwen35-vlm-finetune, levelearn-eval).
Saran saya: (c), tetapi sekarang buat qwen35-vlm-finetune dulu; yang lain menyusul saat tahapnya tiba. Skill kecil lebih hemat konteks dan hanya dimuat saat relevan.

Jawaban: Ikut saran (c). Pisahkan prapemrosesan, fine-tuning, dan evaluasi menjadi skill tersendiri. Sekarang buat qwen35-vlm-finetune terlebih dahulu, dengan menjelaskan format data masukan yang harus tersedia dari tahap prapemrosesan.

B2. Apakah skill juga perlu mode penjelasan untuk dokumen TA — misalnya setiap langkah disertai ringkasan "apa dan mengapa" yang bisa Anda adaptasi ke Bab III?

Jawaban: Ya. Sertakan penjelasan apa yang dilakukan dan mengapa dilakukan untuk membantu dokumentasi TA. Bedakan konfigurasi yang direncanakan, implementasi yang benar-benar dijalankan, dan hasil yang sudah diperoleh. Jangan menulis rencana seolah-olah sudah menjadi hasil eksperimen.

B3. Bahasa isi skill: Indonesia atau Inggris? (Agent bisa membaca keduanya; Inggris sedikit lebih ringkas, Indonesia lebih mudah Anda periksa.)

Jawaban: Bahasa Indonesia. Pertahankan istilah teknis yang umum, singkatan seperti QG dan AE, nama fungsi, serta kode dalam bentuk aslinya.
C. Keputusan teknis training
C1. Pendekatan kode training:
- (a) Script sendiri dengan Transformers + PEFT (+ TRL SFTTrainer bila cocok) — paling transparan, mudah dijelaskan di TA, tetapi lebih banyak ditulis sendiri.
- (b) Memakai framework 2U1/Qwen-VL-Series-Finetune — sudah mendukung Qwen3.5, tetapi Anda bergantung pada kode pihak ketiga.
- (c) Unsloth — cepat dan hemat VRAM, tetapi dukungan Qwen3.5 belum saya verifikasi.
- (d) LLaMA-Factory — konfigurasi YAML/WebUI, dukungan Qwen3.5 belum saya verifikasi.
Saran saya: (a), dengan 2U1 sebagai acuan membaca. Untuk TA, Anda perlu bisa menjelaskan setiap bagian kode.

Jawaban: Ikut saran (a). Gunakan Transformers + PEFT sebagai pendekatan awal, dengan 2U1 sebagai referensi. Gunakan TRL hanya jika kompatibilitas dan penanganan data multimodalnya sudah diverifikasi. Pastikan pengolahan gambar, collator, dan masking label benar.

C2. Platform GPU untuk training/inferensi 9B:
- Colab gratis (T4, 15 GB, tidak mendukung bf16)
- Colab Pro/Pro+ (L4 24 GB / A100 40–80 GB) — Anda berlangganan?
- GPU IT Del (B200) — apakah aksesnya sudah pasti? Lewat SSH/Jupyter?
- Lainnya: ...
Jawaban: Saat ini yang dapat diakses adalah komputer lokal dan Google Colab. GPU IT Del belum dapat diakses sekarang, tetapi akan ada kesempatan mengaksesnya kemudian. Buat skill yang mendukung perpindahan lingkungan tersebut dan tidak bergantung pada GPU institusi untuk pilot awal. Periksa spesifikasi komputer lokal dan GPU Colab aktual sebelum menentukan konfigurasi. Status langganan Colab serta mekanisme akses GPU IT Del belum dikonfirmasi.

C3. Vision encoder: dibekukan (hanya LoRA di bagian bahasa) atau ikut dilatih?
Catatan: README 2U1 menyatakan QLoRA 4-bit tidak bisa digabung dengan melatih bagian vision; melatih vision butuh 16-bit dan GPU lebih besar.

Saran saya: dibekukan untuk pilot dan kemungkinan besar untuk TA.

Jawaban: Bekukan vision encoder untuk pilot dan fokuskan LoRA pada bagian bahasa. Keputusan untuk eksperimen utama dapat dievaluasi kembali berdasarkan hasil pilot dan sumber daya. Gambar tetap harus diproses sebagai masukan multimodal. Perlakukan batasan yang disebut README sebagai batas implementasi framework tersebut, bukan larangan universal.

C4. Mode thinking Qwen3.5: tetap enable_thinking=False seperti notebook Colab Anda?

Saran saya: ya — target latihan adalah soal/pembahasan langsung, tanpa jejak penalaran tersembunyi.

Jawaban: Ya, tetap enable_thinking=False seperti percobaan sebelumnya. AE tetap harus memuat pembahasan lengkap, langkah yang diperlukan, alasan kunci benar, dan alasan setiap pengecoh salah.

C5. Bentuk masukan visual: beberapa gambar frame (sesuai keputusan di dokumen konteks) atau eksplorasi juga input video native model?

Saran saya: gambar frame dulu; video native hanya dicatat sebagai eksplorasi lanjutan.

Jawaban: Ikut saran. Gunakan beberapa gambar frame terurut beserta transkrip dan instruksi tugas terlebih dahulu. Video native dicatat sebagai eksplorasi lanjutan. Jumlah frame, resolusi, dan panjang konteks ditentukan melalui pilot sesuai kapasitas perangkat.
D. Format keluaran (dibutuhkan agar skill bisa memvalidasi hasil)
D1. Apakah keluaran QG dan AE harus berupa JSON? Jika ya, apakah sudah ada skema/contoh dari proposal atau aplikasi LeveLearn (nama field yang dipakai backend)?

Jawaban: Ya, gunakan JSON terstruktur agar dapat divalidasi dan diintegrasikan. Periksa terlebih dahulu apakah backend LeveLearn sudah memiliki skema. Jika belum tersedia, usulkan skema dan validator; jangan menganggap nama field sudah ditetapkan. Pertahankan dua tahap: QG menghasilkan tiga soal per segmen, masing-masing empat opsi dan satu kunci; AE menghasilkan pembahasan per soal.

D2. Bahasa soal dan pembahasan: Indonesia? Bagaimana jika video berbahasa Inggris — soal tetap bahasa Indonesia?

Jawaban: Ya, soal dan pembahasan berbahasa Indonesia, termasuk ketika sumber video berbahasa Inggris. Pertahankan makna, notasi logika, dan istilah teknis yang relevan. Jangan menerjemahkan dengan cara yang mengubah konsep.

D3. Bahasa audio video pilot BDD (hasil transkrip.txt sudah ada): Inggris atau Indonesia?

Jawaban: Belum dikonfirmasi. Periksa audio atau transkrip aktual secara langsung. Jangan menyimpulkan bahasa audio dari bahasa slide. Pastikan juga apakah transkrip.txt benar-benar sudah tersedia sebelum menganggap proses ASR selesai.
E. Dokumen dan administrasi
E1. Dokumen konteks menyebut ada berkas Bab III terbaru (1 November) yang belum diperiksa. Apakah ada keputusan metodologi di sana yang berbeda dari dokumen konteks (mis. konfigurasi LoRA, jumlah frame, model ASR)? Jika berkas bisa ditaruh di repo, sebutkan lokasinya.

Jawaban: Periksa dokumen metodologi terbaru yang tersedia sebelum menetapkan konfigurasi final. Berkas yang tercatat adalah Bab_III_Metodologi_LeveLearn_Lengkap_Revisi.docx, tetapi lokasinya di repo belum dikonfirmasi. Perbedaan keputusan belum dapat dipastikan tanpa membaca isinya. Jika belum dapat diakses, tandai sebagai belum diperiksa dan jangan mengarang konfigurasi LoRA, jumlah frame, atau model ASR.

E2. Setelah skill selesai, perlu saya commit ke git? (Saat ini belum ada yang di-commit.)

Jawaban: Tidak. Jangan pernah melakukan commit maupun push. Seluruh tindakan commit dan push saya lakukan sendiri. Cukup siapkan perubahan, jalankan pemeriksaan yang relevan, dan laporkan file yang diubah beserta hasil pemeriksaannya.

E3. Ada hal lain yang Anda ingin skill ini lakukan atau hindari?

Jawaban:
- Pertahankan checkpoint utama Qwen/Qwen3.5-9B.
- Bedakan percobaan LoRA teks yang sudah dilakukan dari fine-tuning multimodal yang masih perlu disiapkan.
- Pastikan gambar benar-benar diproses sebagai masukan model, bukan hanya mencantumkan nama file.
- Pertahankan QG dan AE sebagai dua tahap pemanggilan; satu model atau adapter dapat digunakan untuk kedua tugas dengan instruksi berbeda.
- Jangan menimpa video, model dasar, atau data asli.
- Pisahkan train, validation, dan test berdasarkan kelompok sumber agar tidak terjadi kebocoran data.
- Jangan menggunakan keluaran AI yang belum diperiksa sebagai jawaban acuan.
- Jalankan pemeriksaan satu batch forward/backward sebelum pelatihan lebih panjang.
- Simpan konfigurasi, versi paket, adapter, log, pembagian data, dan keluaran mentah.
- Uji penyimpanan serta pemuatan ulang adapter.
- Jangan mengklaim peningkatan kualitas hanya berdasarkan penurunan loss latihan.
- Bedakan fakta terverifikasi, keputusan, usulan, dan hal yang belum diketahui.
- Berikan penjelasan bertahap dalam bahasa Indonesia yang sederhana dan jelas.
- Jangan melakukan commit atau push.