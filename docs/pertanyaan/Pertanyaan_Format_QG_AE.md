# Pertanyaan: format prompt & keluaran QG/AE (v0.1)

Tanggal: 2026-11-02. Sumber yang dibandingkan:
- **Riset** = `docs/research/Riset_Prompt_QG_AE_dan_Masukan_VLM_LeveLearn.md`
- **Skill** = `.claude/skills/qwen35-vlm-finetune/references/format-data.md` §2–4 (usulan lama)
- **ADR-0009** (JSON, kunci huruf, metadata disisipkan kode) dan **LeveLearn** (`Assessment.questions` = `[{question, options[], answer, type}]`)

Isi kolom `Jawaban:`. Boleh singkat ("setuju rekomendasi").

## Sudah sepakat (tidak perlu dijawab)
- Keluaran JSON, bahasa Indonesia, `answer_key` huruf A–D; kode yang mengonversi ke format LeveLearn (ADR-0009, Riset §8).
- `segment_id` dan waktu disisipkan kode dari metadata, bukan ditulis model.
- Susunan masukan baseline (Layout A): metadata → frame berlabel waktu → transkrip → instruksi (Skill §4 = Riset §6.2). Tidak diklaim optimal.
- QG dan AE dua panggilan terpisah; QG tidak menghasilkan AE.
- Terverifikasi di template checkpoint: pesan `system` didukung, dan template latih tetap = inferensi (`enable_thinking=False`).

---

## P1. Field per soal QG: `concept` atau `learning_objective`?
- Skill: `concept` (karena proposal V1 Tabel 3.6 meminta "konsep" per soal).
- Riset: `learning_objective` (sasaran belajar singkat).
- Rekomendasi: pakai **`concept`** supaya sejalan dengan proposal; isi berupa konsep yang diuji (mis. "disjungsi", "reduksi tabel kebenaran dengan *").

Jawaban:

## P2. Field `stimulus` di soal QG
Riset menambah `stimulus` (tabel/rumus yang dibutuhkan untuk menjawab). Masalahnya, **LeveLearn saat ini tidak punya tempat untuk stimulus**: hanya `question`, `options`, `answer`. Kalau soal merujuk "tabel tadi", siswa tidak akan melihat tabelnya.
- Opsi a: tetap ada `stimulus`; saat dikonversi ke LeveLearn, kode menggabungkan `stimulus` + `stem` menjadi `question`.
- Opsi b: tanpa `stimulus`; model wajib menulis soal yang berdiri sendiri di `stem`.
- Rekomendasi: **a** (lebih rapi untuk dinilai; konversi sederhana; skema LeveLearn tidak diubah).

Jawaban:

## P3. `status` dan `issue` (penanganan gagal)
Riset menambah `status` ∈ {`ok`, `insufficient_context`, `invalid_question` (AE saja)} + `issue`, supaya model tidak mengarang saat sumber tidak cukup atau kunci salah. Skill lama tidak punya ini.
- Rekomendasi: **pakai**. Konsekuensi: contoh acuan juga perlu sesekali memuat status gagal, kalau tidak model tidak pernah belajar memakainya (untuk pilot cukup dicatat).

Jawaban:

## P4. `evidence_ids` (ID frame/transkrip pendukung)
Riset meminta model mencantumkan ID bukti (`F001`, `T001`). Ini berarti setiap frame dan potongan transkrip diberi ID di prompt. Validator mengecek ID-nya ada, tapi **tidak** membuktikan isinya benar.
- Rekomendasi: **pakai**, ID = nomor frame (`F001`–`F027`) dan nomor potongan transkrip (`T01`–`T15`). Membantu penilai melacak sumber soal.

Jawaban:

## P5. Bentuk pembahasan AE
- Skill lama: `correct_explanation` (satu teks) + `distractor_explanations` (dict) + `enrichment` (pengayaan berlabel).
- Riset: `principle` → `steps[]` → `conclusion` + `distractor_explanations[]` + `evidence_ids`; **tanpa** pengayaan di v0.1.
- Rekomendasi: **ikuti Riset** (lebih terstruktur, sesuai worked example; mudah dinilai per bagian). Pengayaan ditunda.

Jawaban:

## P6. Konvensi waktu di masukan
Riset meminta satu konvensi eksplisit. Skill memakai detik absolut (`t_abs`); `cek_sampel.py` sementara menulis keduanya.
- Rekomendasi: **detik absolut video** di masukan (250–387 dtk), karena pembahasan AE nanti merujuk waktu video asli. Relatif tetap disimpan di manifest.

Jawaban:

## P7. Penulis contoh acuan pilot
Contoh latih harus `diperiksa` manusia. Siapa yang menulis/memeriksa 3 soal + 3 pembahasan untuk segmen pilot?
- Opsi a: Anda menulis sendiri dari nol.
- Opsi b: saya buat **draf** (status `draf_ai`), Anda periksa dan koreksi → `diperiksa`.
- Catatan: contoh di Riset §9 (makna tanda `*`) bisa jadi salah satu soal, setelah dicek terhadap frame aktual.

Jawaban:




Jawaban seluruh nya :
P1. Field per soal QG: concept atau learning_objective?
Jawaban: Gunakan concept untuk mencatat konsep yang diuji, misalnya “disjungsi” atau “peringkasan tabel kebenaran”. concept dan learning_objective memiliki makna berbeda, sehingga jangan sekadar mengganti nama field tetapi tetap mengisinya dengan kalimat sasaran belajar. Field sasaran belajar dapat ditambahkan kemudian jika diperlukan.
P2. Field stimulus di soal QG
Jawaban: Pilih opsi a. Pertahankan stimulus terpisah dari stem pada data penelitian. Saat dikonversi ke LeveLearn, kode menggabungkan keduanya menjadi question.

Pastikan stimulus benar-benar terbaca pada tampilan mahasiswa. Untuk v0.1, gunakan stimulus teks atau representasi tabel yang didukung tampilan. Jika gambar/tabel belum dapat ditampilkan, soal harus menyediakan informasi yang cukup dalam teks. Menggabungkan nama file gambar dengan stem saja tidak cukup.
P3. status dan issue — penanganan gagal
Jawaban: Gunakan keduanya:
- QG: ok atau insufficient_context.
- AE: ok, insufficient_context, atau invalid_question.
- issue: null ketika berhasil; penjelasan masalah ketika gagal.
Untuk QG, ok mensyaratkan tiga soal sesuai kontrak. Jika sumber tidak mencukupi, jangan mengarang soal tambahan; jumlah yang kurang tetap dicatat sebagai kegagalan memenuhi target.

Sertakan contoh kegagalan yang diperiksa manusia dalam data latihan ketika dataset mulai disusun. Tanpa contoh tersebut, perilaku pelaporan gagal belum dilatih secara khusus—bukan berarti model sama sekali tidak mungkin menggunakannya melalui instruksi.

Validator tetap memeriksa hasil secara independen; status="ok" dari model tidak otomatis membuktikan keluaran valid.
P4. evidence_ids — ID frame/transkrip pendukung
Jawaban: Gunakan untuk membantu keterlacakan. Seragamkan format menjadi F001, F002, … untuk frame dan T001, T002, … untuk potongan transkrip.

Jumlah ID mengikuti data aktual, bukan ditetapkan selalu 27 frame atau 15 potongan transkrip. ID harus unik dalam segmen dan dipasangkan dengan segment_id oleh kode.

Validator memeriksa keberadaan ID dan kesesuaian segmennya. Pemeriksaan apakah bukti tersebut benar-benar mendukung isi soal atau pembahasan tetap dilakukan penilai. Ini bukan tugas temporal grounding baru.
P5. Bentuk pembahasan AE
Jawaban: Ikuti struktur riset:
- principle: aturan atau konsep yang digunakan.
- steps[]: langkah penerapan yang diperlukan.
- conclusion: simpulan mengapa kunci benar.
- distractor_explanations[]: alasan setiap pengecoh salah, dengan option_id dan reason.
- evidence_ids: rujukan bukti pendukung.
Gunakan array untuk penjelasan pengecoh dan pastikan tepat mencakup tiga opsi selain kunci.

Pengayaan ditunda pada v0.1. Struktur ini merupakan adaptasi prinsip penjelasan dan worked examples, bukan format JSON yang telah dibuktikan paling unggul oleh jurnal. Jumlah langkah mengikuti kebutuhan soal.

Jika AE menemukan kunci salah atau tidak tunggal, gunakan invalid_question; jangan memaksakan pembenaran.
P6. Konvensi waktu di masukan
Jawaban: Gunakan detik absolut video pada masukan model, dengan label konvensi yang eksplisit.

Untuk pilot:
- Awal: 250 detik.
- Akhir: 387 detik.
- Durasi: 137 detik.
Waktu relatif tetap disimpan di manifest untuk kebutuhan pemrosesan. Kode melakukan konversi relatif–absolut dan memasang rujukan waktu pada keluaran. Model tidak diminta menebak timestamp.
P7. Penulis contoh acuan pilot
Jawaban: Gunakan alur opsi b: Claude menyusun tiga soal dan tiga pembahasan sebagai draf AI, kemudian manusia memeriksa serta mengoreksinya.

Status awal harus draf_ai. Jangan mengubahnya menjadi diperiksa sebelum pemeriksaan benar-benar dilakukan. Identitas pemeriksa dan hasil pemeriksaannya dicatat; pemeriksa yang akan digunakan belum ditetapkan dalam keputusan ini.

Contoh tentang tanda * boleh menjadi kandidat setelah dicocokkan dengan frame dan transkrip aktual. Satu segmen pilot cukup untuk memeriksa alur, tetapi belum cukup sebagai dataset fine-tuning utama.