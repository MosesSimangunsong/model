"""Pecah Data/Pilot/contoh_acuan_pilot.json -> manifest segmen + contoh latih JSONL.

Keluaran (format-data.md §1 dan §3):
  Data/Pilot/segmen.json        manifest segmen (frame, transkrip, waktu)
  Data/Pilot/contoh_latih.jsonl 1 contoh QG + 3 contoh AE (satu baris per contoh)

Target tiap contoh divalidasi dengan prompt_qg_ae sebelum ditulis. Status anotasi disalin
apa adanya dari berkas acuan; hanya `diperiksa` yang boleh masuk train/val/test.

Pakai (dari root repo, venv aktif):
  python scripts/bangun_contoh_latih.py
"""
import json
import re
from pathlib import Path

from prompt_qg_ae import INSTRUKSI_AE, INSTRUKSI_QG, baca_transkrip, validasi_ae, validasi_qg

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "Data" / "Pilot"
ACUAN = PILOT / "contoh_acuan_pilot.json"
OFFSET_ABS = 250
FRAME_INTERVAL = 5
END_ABS = 387
SOURCE_GROUP = "pilot_bdd_source_01"
VIDEO_ID = "pilot_bdd_01"


def bangun_manifest(segment_id):
    frames = []
    for p in sorted((PILOT / "frames").glob("frame_*.jpg")):
        n = int(re.search(r"(\d+)", p.stem).group(1))
        rel = (n - 1) * FRAME_INTERVAL
        frames.append({"id": f"F{n:03d}", "path": p.relative_to(ROOT).as_posix(),
                       "t_rel": rel, "t_abs": rel + OFFSET_ABS})
    transkrip = [{"id": t["id"], "t_rel_start": t["t_abs_start"] - OFFSET_ABS,
                  "t_rel_end": t["t_abs_end"] - OFFSET_ABS, "text": t["text"]}
                 for t in baca_transkrip(PILOT / "transkrip.txt", OFFSET_ABS)]
    return {
        "video_id": VIDEO_ID, "segment_id": segment_id, "source_group_id": SOURCE_GROUP,
        "source_file": "Data/Pilot/segmen_0410_0627.mp4",
        "start_seconds": OFFSET_ABS, "end_seconds": END_ABS,
        "frames": frames,
        "frame_sampling": {"method": "fps", "value": "1/5", "tool": "ffmpeg"},
        "transcript": {"asr_model": "faster-whisper small (cpu)", "language": "id",
                       "status": "diperiksa", "catatan": "dikoreksi manual oleh pengguna",
                       "segments": transkrip},
        "ocr": None,
    }


def main():
    acuan = json.loads(ACUAN.read_text(encoding="utf-8"))
    segment_id = acuan["segment_id"]
    manifest = bangun_manifest(segment_id)
    id_valid = {f["id"] for f in manifest["frames"]} | {t["id"] for t in manifest["transcript"]["segments"]}
    frame_paths = [f["path"] for f in manifest["frames"]]
    soal_list = acuan["qg"]["questions"]

    galat = validasi_qg(acuan["qg"], id_valid)
    for ae, soal in zip(acuan["ae"], soal_list):
        galat += [f"{soal['id']}: {e}" for e in validasi_ae(ae, soal, id_valid)]
    assert [a["question_id"] for a in acuan["ae"]] == [q["id"] for q in soal_list], "urutan AE != urutan soal"
    if galat:
        raise SystemExit("Target tidak lolos validasi, tidak ada berkas ditulis:\n- " + "\n- ".join(galat))

    dasar = dict(split=None, source_group_id=SOURCE_GROUP,
                 segment_manifest=(PILOT / "segmen.json").relative_to(ROOT).as_posix(),
                 frame_paths=frame_paths, annotation_status=acuan["annotation_status"],
                 annotator=acuan.get("pemeriksa"))
    baris = [dict(dasar, example_id=f"{segment_id}_qg", task="qg", instruction=INSTRUKSI_QG,
                  input_extra=None, target=json.dumps(acuan["qg"], ensure_ascii=False))]
    for ae, soal in zip(acuan["ae"], soal_list):
        baris.append(dict(dasar, example_id=f"{segment_id}_ae_{soal['id']}", task="ae",
                          instruction=INSTRUKSI_AE, input_extra=soal,
                          target=json.dumps(ae, ensure_ascii=False)))

    (PILOT / "segmen.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(PILOT / "contoh_latih.jsonl", "w", encoding="utf-8") as f:
        for b in baris:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    print(f"segmen.json: {len(manifest['frames'])} frame, {len(manifest['transcript']['segments'])} potongan transkrip")
    print(f"contoh_latih.jsonl: {len(baris)} contoh -> " + ", ".join(b["example_id"] for b in baris))
    print(f"annotation_status = {acuan['annotation_status']} | pemeriksa = {acuan.get('pemeriksa')}")
    if acuan["annotation_status"] != "diperiksa":
        print("PERINGATAN: status bukan 'diperiksa' -> belum boleh masuk train/val/test.")


if __name__ == "__main__":
    main()
