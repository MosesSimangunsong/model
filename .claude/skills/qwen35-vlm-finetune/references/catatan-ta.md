# Panduan catatan untuk dokumen TA

Setiap langkah ditutup dengan blok berikut. Bahasanya sederhana, dan istilah teknis dijelaskan saat pertama muncul.

```markdown
### <Nama langkah>
**Status:** Rencana | Diimplementasikan | Hasil | Belum diketahui
**Apa:** <yang dilakukan, 1–3 kalimat>
**Mengapa:** <alasan metodologis, 1–3 kalimat>
**Bukti:** <path log/file/keluaran; kosong jika status Rencana>
**Catatan reproduksi:** <versi paket, seed, parameter penting>
```

## Contoh (status sesuai keadaan per 2 November 2026)

### Verifikasi modul target LoRA
**Status:** Diimplementasikan (pembacaan konfigurasi checkpoint, tanpa memuat bobot)
**Apa:** Struktur lapisan Qwen3.5-9B dibaca dari `config.json` dan bentuk tensor dari header safetensors. Hasilnya: bagian bahasa terdiri atas 24 lapisan linear attention dan 8 lapisan full attention.
**Mengapa:** Pemilihan modul target LoRA menentukan bagian model yang dapat beradaptasi. Konfigurasi sebelumnya ternyata hanya mengenai 8 dari 32 lapisan.
**Bukti:** Jumlah parameter terhitung (8 × 69.632 = 557.056) cocok dengan laporan notebook.
**Catatan reproduksi:** Checkpoint lokal `Model/Qwen3.5-9B`; dibaca 2 November 2026.

### Pembekuan vision encoder
**Status:** Rencana
**Apa:** Bobot vision encoder dibekukan; adapter LoRA hanya dipasang pada bagian bahasa. Frame tetap diproses oleh vision encoder dan memengaruhi prediksi.
**Mengapa:** Menghemat memori dan sesuai dengan kuantisasi 4-bit. Keputusan untuk eksperimen utama dievaluasi kembali setelah pilot.
**Bukti:** —

## Kalimat yang aman vs tidak aman

| Hindari | Gunakan |
|---|---|
| "Fine-tuning meningkatkan kualitas soal." (tanpa evaluasi) | "Loss latih turun dari X ke Y; dampak pada kualitas belum dievaluasi." |
| "Model memahami video." | "Model menerima N frame terurut dan transkrip ASR sebagai masukan." |
| "Seluruh model dilatih." | "Hanya adapter LoRA (P parameter, Q% dari total) yang diperbarui; bobot dasar dibekukan." |
| "Whisper menghasilkan transkrip akurat." | "Transkrip dihasilkan Whisper `small` lalu diperiksa manual." (hanya jika sudah diperiksa) |
