# ADR-0003: QLoRA 4-bit untuk pilot

- **Status:** Sementara
- **Tanggal:** 2026-11-02
- **Pemicu tinjau ulang:** **GPU IT Del (B200) sudah dapat diakses**, atau pilot menunjukkan kuantisasi merusak kualitas/kompatibilitas.

## Konteks
Sumber daya yang tersedia saat ini hanya laptop (RTX 3060, 6 GB) dan Google Colab (jenis GPU belum dikonfirmasi). Di Colab T4, model 4-bit sudah memakai ~7,35 GiB setelah dimuat, sebelum aktivasi gambar. Proposal V1 §3.3.3 menulis "LoRA menjadi kandidat awal; QLoRA dipertimbangkan jika dukungan implementasi dan kebutuhan memori sesuai".

## Keputusan
Untuk **pilot**, gunakan **QLoRA**: bobot dasar dikuantisasi 4-bit (NF4, double quantization) dan dibekukan; hanya adapter LoRA yang dilatih.

**Keputusan ini dapat berubah** setelah GPU IT Del tersedia. Pada saat itu, LoRA dengan bobot dasar 16-bit (bf16) dipertimbangkan kembali untuk eksperimen utama, karena kuantisasi bukan pilihan metodologis melainkan penyesuaian terhadap keterbatasan memori.

## Alasan
- Model 9B dalam presisi penuh tidak muat pada GPU yang tersedia sekarang.
- Kuantisasi menghemat memori, tetapi **bukan** bukti peningkatan ketepatan (proposal V1 §2.4.4).

## Alternatif
- LoRA 16-bit sekarang: tidak muat pada GPU yang tersedia.
- Model yang lebih kecil: bertentangan dengan ADR-0001.

## Konsekuensi
- Jika kemudian beralih ke LoRA 16-bit, eksperimen utama harus **dijalankan ulang** dengan konfigurasi yang sama untuk semua kondisi (M0/M1 × V/T); hasil pilot QLoRA tidak boleh dicampur dengan hasil uji 16-bit.
- Laporkan presisi bobot dasar (4-bit atau 16-bit) di setiap log dan di dokumen TA.
- Menurut README 2U1, framework mereka tidak mendukung QLoRA bersamaan dengan pelatihan bagian vision — relevan bila ADR-0004 berubah.
- **Tindakan dokumen:** kalimat §3.3.3 proposal diselaraskan agar menyatakan QLoRA untuk pilot dan peluang beralih ke LoRA 16-bit pada GPU IT Del.

## Rujukan
- Jawaban pengguna 2 November 2026; `Pertanyaan_Skill_VLM.md` C2–C3; proposal V1 §2.4.4, §3.3.3; README 2U1 "Training Notes" (commit `6707820`).
