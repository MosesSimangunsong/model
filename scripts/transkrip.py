"""Buat transkrip ucapan dari segmen pilot memakai faster-whisper.

Pemakaian:
    python scripts/transkrip.py                 # bahasa dideteksi otomatis
    python scripts/transkrip.py --bahasa en     # paksa bahasa (en / id)
    python scripts/transkrip.py --model medium  # model lebih besar
"""
import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "Data" / "Pilot" / "segmen_0410_0627.mp4"
OUT_JSON = ROOT / "Data" / "Pilot" / "transkrip.json"
OUT_TXT = ROOT / "Data" / "Pilot" / "transkrip.txt"
OFFSET_ABSOLUT = 250  # detik awal segmen pada video penuh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--bahasa", default=None)
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    args = ap.parse_args()

    model = WhisperModel(args.model, device=args.device, compute_type="int8")
    segments, info = model.transcribe(str(VIDEO), language=args.bahasa, vad_filter=True)
    print(f"Bahasa terdeteksi: {info.language} (peluang {info.language_probability:.2f})")

    hasil = []
    for s in segments:
        hasil.append({
            "start_relatif": round(s.start, 2),
            "end_relatif": round(s.end, 2),
            "start_absolut": round(s.start + OFFSET_ABSOLUT, 2),
            "end_absolut": round(s.end + OFFSET_ABSOLUT, 2),
            "teks": s.text.strip(),
        })

    OUT_JSON.write_text(
        json.dumps({"bahasa": info.language, "model": args.model, "segmen": hasil},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_TXT.write_text(
        "\n".join(f"[{h['start_relatif']:6.1f}-{h['end_relatif']:6.1f}] {h['teks']}" for h in hasil),
        encoding="utf-8",
    )
    print(f"{len(hasil)} potongan ucapan disimpan ke {OUT_TXT}")


if __name__ == "__main__":
    main()
