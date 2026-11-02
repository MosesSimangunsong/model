# ADR-0015: Kontrak prompt dan JSON QG/AE v0.1

- **Status:** Sementara (v0.1 untuk pilot)
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Pilot parsing/penilaian menunjukkan field tidak terpakai atau sulit diisi; tampilan LeveLearn berubah (mis. bisa menampilkan tabel); keputusan perubahan skema LeveLearn untuk AE.

## Konteks
ADR-0009 menetapkan JSON, kunci huruf A–D, dan metadata waktu dari kode, tetapi nama field masih usulan. Riset `docs/research/Riset_Prompt_QG_AE_dan_Masukan_VLM_LeveLearn.md` (§4–§8) mengusulkan isi AE, kriteria soal, susunan masukan, dan kontrak JSON yang berbeda dari usulan lama di `format-data.md`. Perbedaan diputuskan pengguna lewat `Pertanyaan_Format_QG_AE.md` (P1–P7).

## Keputusan
1. **QG** per soal: `id`, `concept`, `stimulus`, `stem`, `options` (A–D), `answer_key`, `evidence_ids`. `concept` berisi konsep yang diuji (mis. "disjungsi"), bukan kalimat sasaran belajar.
2. **Stimulus** terpisah dari `stem` di data penelitian; saat konversi ke LeveLearn, kode menggabungkan `stimulus` + `stem` menjadi `question`. Stimulus harus berupa teks/tabel teks yang benar-benar terbaca siswa; nama file gambar tidak cukup.
3. **Status:** QG ∈ {`ok`, `insufficient_context`}; AE ∈ {`ok`, `insufficient_context`, `invalid_question`}; `issue` = null bila berhasil. QG `ok` mensyaratkan tepat 3 soal; kekurangan dicatat sebagai kegagalan target. Validator memeriksa secara independen dari `status`.
4. **evidence_ids:** `F001…` untuk frame, `T001…` untuk potongan transkrip; jumlah mengikuti data aktual, unik dalam segmen. Validator hanya memeriksa keberadaan ID; dukungan isi dinilai penilai.
5. **AE:** `question_id`, `principle`, `steps[]` (`order`, `statement`), `conclusion`, `distractor_explanations[]` (`option_id`, `reason`, tepat 3 opsi selain kunci), `evidence_ids`. Tanpa pengayaan di v0.1. Kunci salah/tidak tunggal → `invalid_question`.
6. **Waktu:** masukan memakai detik absolut video dengan label konvensi eksplisit (pilot 250–387 dtk); relatif tetap di manifest; kode memasang rujukan waktu pada keluaran.
7. **Susunan pesan:** `system` = aturan tetap (Riset §7.1); `user` = metadata → frame berlabel ID+waktu → transkrip berlabel ID+waktu → instruksi tugas (Layout A, Riset §6.2). Tidak diklaim optimal.
8. **Contoh acuan pilot:** Claude menyusun draf (`draf_ai`), manusia memeriksa; status `diperiksa` hanya setelah pemeriksaan nyata, dengan identitas pemeriksa dan hasil dicatat.

## Alasan
Struktur AE mengadaptasi penjelasan kunci/pengecoh dan worked examples (Riset R1–R3); kriteria QG dari R4–R7, R12. JSON adalah kontrak rekayasa untuk validasi, bukan format yang dibuktikan unggul secara pedagogis. `concept` dipertahankan agar sejalan dengan proposal V1 Tabel 3.6.

## Alternatif yang dipertimbangkan
- `learning_objective` menggantikan `concept` — ditolak; maknanya berbeda, dapat ditambah kemudian.
- Tanpa `stimulus`, soal wajib berdiri sendiri di `stem` — ditolak; stimulus terpisah lebih mudah dinilai.
- AE satu teks (`correct_explanation`) + `enrichment` (usulan lama) — ditolak untuk v0.1.

## Konsekuensi
- `format-data.md` §2–4 diperbarui mengikuti ADR ini.
- Contoh latih kegagalan (`insufficient_context`/`invalid_question`) yang diperiksa manusia perlu disertakan saat dataset disusun; tanpa itu perilaku melapor gagal belum dilatih khusus.
- Satu segmen pilot hanya untuk memeriksa alur, bukan dataset fine-tuning.

## Rujukan
- `Pertanyaan_Format_QG_AE.md` (jawaban P1–P7); Riset §4–§8; ADR-0006, ADR-0009, ADR-0010.
