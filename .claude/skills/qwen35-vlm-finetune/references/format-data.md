# Kontrak data: masukan dari prapemrosesan dan contoh latih

Skema manifest, keluaran model, dan contoh latih di sini berstatus **usulan**. Format penyimpanan backend LeveLearn sudah diperiksa (bagian 2a) dan menjadi batasan.

## 0. Format soal di LeveLearn yang sudah ada (terverifikasi 2 November 2026)

Sumbernya `src/` (salinan repo LeveLearn):
- `Backend-LeveLearn-main/.../prisma/schema.prisma`: model `Assessment { chapterId, instruction, questions Json?, answers Json? }`. Assessment terikat ke `Chapter`. **Tidak ada** model video, segmen, atau timestamp.
- `questions` disimpan sebagai **string JSON** (hasil `JSON.stringify`) berisi array. Bentuk setiap elemen ada di `FrontendWeb-.../src/dto/AssessmentDto.ts` dan `prisma/seed.js`:
  ```json
  {"question": "…", "options": ["…", "…", "…", "…"], "answer": "<teks opsi yang benar>", "type": "MC"}
  ```
  `type` bernilai `"MC"` (pilihan ganda) atau `"EY"` (esai). `options` tidak dipakai untuk esai.
- **`answer` berisi teks opsi, bukan huruf.** Mobile (`lib/service/chapter_service.dart`) memetakan `options`→`option` dan `answer`→`correctedAnswer`, lalu mencocokkan jawaban dengan perbandingan teks (`trim().toLowerCase()`). Huruf A/B/C… hanya ditampilkan di web berdasarkan urutan (`optionList.charAt(index)`).
- Backend (`AssessmentService.js`) meneruskan data tanpa validasi isi.
- **Tidak ada field pembahasan (AE)** di skema maupun UI.

Konsekuensinya untuk proyek ini:
- Hasil QG dapat dikonversi ke format di atas. Opsi harus unik setelah `trim().toLowerCase()`, karena kunci dicocokkan lewat teks.
- Menyimpan pembahasan AE, ID segmen, dan waktu segmen **membutuhkan perubahan skema atau field baru** di LeveLearn. Itu **keputusan terbuka** di luar skill ini. Jangan menganggap field-nya sudah ada. Proposal V1 Tabel 3.7 mengusulkan entitas `VideoSource/VideoSegment`, `GeneratedQuestion`, `GeneratedExplanation`, `SegmentReference`, `ReviewRecord`, dan lainnya. Statusnya usulan, belum ada di `schema.prisma`.
- Seed yang ada berisi soal True/False dan 5 opsi. Format QG proyek (tepat 4 opsi) adalah keputusan penelitian, bukan batasan backend.

## 1. Manifest segmen (keluaran skill prapemrosesan)

Satu file JSON per segmen, misalnya `Data/<video_id>/<segment_id>/segmen.json`:

```json
{
  "video_id": "pilot_bdd_01",
  "segment_id": "pilot_bdd_01_s001",
  "source_group_id": "pilot_bdd_source_01",
  "source_file": "Data/Video/<nama asli>.mp4",
  "start_seconds": 250,
  "end_seconds": 387,
  "frames": [
    {"path": "Data/Pilot/frames/frame_001.jpg", "t_rel": 0, "t_abs": 250}
  ],
  "frame_sampling": {"method": "fps", "value": "1/5", "tool": "ffmpeg <versi>"},
  "transcript": {
    "asr_model": "faster-whisper small int8",
    "language": "id",
    "status": "mentah | diperiksa",
    "segments": [{"t_rel_start": 0.0, "t_rel_end": 10.0, "text": "..."}]
  },
  "ocr": null
}
```

Wajib:
- Bedakan `t_rel` (relatif segmen) dan `t_abs` (absolut video).
- `transcript.status` harus `diperiksa` sebelum transkrip dipakai untuk data latih final. Untuk pilot, transkrip `mentah` boleh dipakai asalkan dicatat. Contoh dari pilot: Whisper menulis "Knya" padahal kemungkinan maksudnya "Q-nya", sehingga koreksi manual memang perlu.
- `ocr` dipakai hanya untuk kondisi pembanding. OCR bukan transkrip ucapan.

## 2. Keluaran yang diharapkan model (kontrak v0.1, ADR-0015)

Model **tidak** menulis `segment_id` atau waktu; kode memasangnya dari manifest (`0 ≤ start < end ≤ durasi video`, detik video penuh, proposal V1 §3.5.2).

**QG**: satu panggilan per segmen.
```json
{
  "schema_version": "qg-0.1",
  "status": "ok",
  "questions": [
    {
      "id": "q1",
      "concept": "…konsep yang diuji, mis. disjungsi…",
      "stimulus": "…tabel/rumus teks yang dibutuhkan, atau \"\"…",
      "stem": "…",
      "options": {"A": "…", "B": "…", "C": "…", "D": "…"},
      "answer_key": "B",
      "evidence_ids": ["F014", "T004"]
    }
  ],
  "issue": null
}
```
Validasi: `status` ∈ {`ok`, `insufficient_context`}; bila `ok`, tepat 3 soal dan `issue` null; tiap soal: 4 opsi A–D tidak kosong dan tidak duplikat, `answer_key` ∈ {A,B,C,D}, `concept` dan `stem` terisi, `evidence_ids` tidak kosong dan semuanya ada di segmen. Validator memeriksa secara independen dari `status`.

**AE**: satu panggilan per soal (masukan: konteks segmen + satu soal QG beserta kunci).
```json
{
  "schema_version": "ae-0.1",
  "status": "ok",
  "question_id": "q1",
  "principle": "…aturan yang dipakai…",
  "steps": [{"order": 1, "statement": "…"}],
  "conclusion": "…mengapa kunci benar…",
  "distractor_explanations": [
    {"option_id": "A", "reason": "…"},
    {"option_id": "C", "reason": "…"},
    {"option_id": "D", "reason": "…"}
  ],
  "evidence_ids": ["F014", "T004"],
  "issue": null
}
```
Validasi: `status` ∈ {`ok`, `insufficient_context`, `invalid_question`}; bila `ok`: `principle`, `conclusion`, ≥1 `steps` terisi, `distractor_explanations` tepat mencakup 3 opsi selain kunci, `evidence_ids` valid. Kunci salah/tidak tunggal → `invalid_question`, bukan pembenaran. Tanpa pengayaan di v0.1.

### 2a. Konversi ke format LeveLearn (usulan)

Model menghasilkan `answer_key` berupa huruf. Kode lalu mengonversinya:

```python
def ke_levelearn(q):
    opsi = [q["options"][k] for k in "ABCD"]
    norm = [o.strip().lower() for o in opsi]
    assert len(set(norm)) == 4, "opsi duplikat: kunci berbasis teks jadi ambigu"
    question = (q["stimulus"].strip() + "\n\n" + q["stem"]).strip()  # ADR-0015: stimulus + stem
    return {"question": question, "options": opsi, "answer": q["options"][q["answer_key"]], "type": "MC"}
```

Alasan memakai huruf, bukan teks: model tidak perlu menyalin ulang teks opsi persis, sehingga risiko kunci tidak cocok dengan opsi berkurang. Ini usulan; alternatifnya model langsung menghasilkan format backend. Catat pilihan yang diambil.

## 3. Contoh latih (JSONL, satu baris per contoh)

```json
{
  "example_id": "pilot_bdd_01_s001_qg",
  "task": "qg",
  "split": "train",
  "source_group_id": "pilot_bdd_source_01",
  "segment_manifest": "Data/Pilot/segmen.json",
  "frame_paths": ["Data/Pilot/frames/frame_001.jpg"],
  "instruction": "…",
  "input_extra": null,
  "target": "{…JSON QG atau AE sebagai string…}",
  "annotation_status": "diperiksa",
  "annotator": "…"
}
```

- Untuk AE, `input_extra` berisi soal, opsi, dan kunci (dari target QG yang sudah diperiksa).
- `annotation_status` ∈ {`draf_ai`, `draf_manusia`, `diperiksa`}. **Hanya `diperiksa` yang boleh masuk train/val/test.**
- Satu segmen menghasilkan 1 contoh QG + 3 contoh AE. Keempatnya **satu sumber**, bukan empat sumber independen.

## 4. Susunan pesan (user content)

Layout A (ADR-0015, Riset §6.2): `system` = aturan tetap; `user` = metadata → frame berlabel ID + detik absolut → transkrip berlabel ID + detik absolut → instruksi tugas. Implementasi: `scripts/prompt_qg_ae.py` (`susun_pesan`).

```text
[system] aturan tetap (Riset §7.1)
[user]   Segmen pilot_bdd_01_s001 | waktu: detik absolut video | 250–387 dtk
         Frame F001 | 250 dtk  <gambar>
         Frame F002 | 255 dtk  <gambar> ...
         Transkrip:
         T001 [250.0–260.0] ...
         Tugas: <instruksi QG, atau instruksi AE + soal JSON>
```

Susunan ini harus **identik** antara training dan inferensi. Simpan fungsinya di satu tempat dan pakai di keduanya. Pesan `system` sudah diuji pada template checkpoint: render penuh = prompt + target (`enable_thinking=False`).

Fungsi yang sama juga menerima parameter kondisi, misalnya `pakai_frame: bool` dan `pakai_ocr: bool`, untuk kondisi -V/-T dan transkrip+OCR dalam desain eksperimen. Selain blok frame atau OCR, teks lain (transkrip, waktu, instruksi) harus sama persis di setiap kondisi.

## 5. Split

- Sasaran proposal V1: sekitar 70% latih, 15% validasi, 15% uji, dihitung per kelompok sumber. Angka aktual mengikuti ukuran kelompok.
- Bagi berdasarkan `source_group_id` **sebelum** membentuk contoh. Semua frame, varian prompt, QG, dan AE dari sumber yang sama berada di split yang sama.
- Simpan manifest split (`splits.json`: daftar `source_group_id` per split + seed).
- Data yang dipakai menyetel pilot tidak dipakai lagi untuk klaim uji akhir.
- Target perencanaan (≥40 video atau 300 segmen dari ≥15 sumber) bukan jumlah data yang tersedia.
