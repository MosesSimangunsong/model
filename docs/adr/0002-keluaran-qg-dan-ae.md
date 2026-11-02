# ADR-0002: Keluaran hanya QG dan AE, dalam dua tahap

- **Status:** Diterima
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** Keputusan baru pengguna/pembimbing mengenai keluaran tambahan.

## Konteks
Proposal V1 (28 Oktober) menyebut tiga keluaran: *materi penjelasan*, soal beserta kunci dan pembahasan, serta rentang waktu bukti video. Catatan keputusan 31 Oktober menyatakan ringkasan bukan tugas keluaran utama dan waktu segmen hanyalah metadata. Pengguna mengonfirmasi pada 2 November 2026 bahwa keluaran **hanya dua**; proposal belum diperbarui.

## Keputusan
Model menghasilkan dua jenis keluaran:
1. **QG (Question Generation):** satu pemanggilan per segmen → 3 soal pilihan ganda, masing-masing 4 opsi dan tepat 1 kunci.
2. **AE (Answer Explanation):** satu pemanggilan per soal → pembahasan (alasan kunci benar, langkah, alasan setiap pengecoh salah).

Satu checkpoint/adapter boleh dipakai untuk keduanya dengan instruksi berbeda. Waktu awal–akhir segmen adalah **metadata yang disalin kode**, bukan keluaran model. "Materi penjelasan"/ringkasan **tidak** menjadi tugas model.

## Alasan
- Fokus penelitian pada soal dan pembahasan; menambah tugas memperbesar kebutuhan anotasi dan evaluasi.
- Memisahkan QG dan AE membuat AE dapat dievaluasi per soal dan kegagalan satu tahap tidak merusak tahap lain.

## Konsekuensi
- Tiga soal valid = 1 pemanggilan QG + 3 pemanggilan AE (di luar retry).
- Data latih: satu segmen menghasilkan 1 contoh QG + 3 contoh AE dari satu sumber.
- **Tindakan dokumen:** Bab I–III proposal perlu disesuaikan (pendahuluan Bab III masih menyebut "tiga keluaran yang dinilai setara").

## Rujukan
- Jawaban pengguna 2 November 2026; `Konteks_Fine_Tuning_Qwen35_9B_LeveLearn_untuk_Claude.md` §3–4; proposal V1 §3.5.2 dan Tabel 3.6.
