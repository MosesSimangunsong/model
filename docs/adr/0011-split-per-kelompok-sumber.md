# ADR-0011: Pembagian data per kelompok sumber

- **Status:** Diterima
- **Tanggal:** 2026-11-02

## Keputusan
- Train/validasi/uji dibagi berdasarkan `source_group_id` **sebelum** membentuk contoh; sasaran sekitar 70/15/15 (proposal V1 §3.4.2).
- Semua frame, varian prompt, contoh QG, dan AE dari satu sumber berada di split yang sama; seri dengan slide berulang dikelompokkan.
- Data pilot berasal dari bagian pengembangan dan tidak dipakai untuk uji akhir.
- Daftar split dan seed disimpan dan tidak diubah karena hasil uji.
- Hanya contoh berstatus `diperiksa` (diperiksa manusia) yang menjadi target.

## Alasan
Mencegah kebocoran data: segmen bertetangga atau slide yang sama di train dan test akan menggelembungkan hasil.

## Rujukan
- Proposal V1 §3.3.1, §3.4.2; dokumen konteks §7.3.
