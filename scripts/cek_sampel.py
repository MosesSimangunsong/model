"""Cek 1 contoh multimodal (frame + transkrip + instruksi) tanpa memuat model 9B.

Hanya memuat processor. Yang diperiksa:
  1. token visual per frame pada beberapa max_side (jawab ADR-0006: ~1.900 token/frame hanya perkiraan)
  2. jumlah token gambar di input_ids == image_grid_thw / merge^2 (gambar benar-benar masuk)
  3. label prompt = -100, label hanya pada jawaban assistant
  4. template latih (prefix + jawaban) == template inferensi

Target jawaban di sini DUMMY, hanya untuk uji mekanik. Bukan contoh acuan QG/AE.

Pakai (dari root repo, venv aktif):
  python scripts/cek_sampel.py
  python scripts/cek_sampel.py --stride 2 --max-side 1024
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from prompt_qg_ae import baca_transkrip, susun_pesan

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "Model" / "Qwen3.5-9B"
FRAMES = sorted((ROOT / "Data" / "Pilot" / "frames").glob("frame_*.jpg"))
TRANSKRIP = ROOT / "Data" / "Pilot" / "transkrip.txt"
OFFSET_ABS = 250  # detik absolut awal segmen (AGENTS.md)
FRAME_INTERVAL = 5  # fps=1/5
TARGET_DUMMY = '{"catatan": "DUMMY untuk uji mekanik, bukan contoh acuan"}'
SEGMENT = {"segment_id": "pilot_bdd_01_s001", "start": 250, "end": 387}
SOAL_DUMMY = {"id": "q1", "concept": "DUMMY", "stimulus": "", "stem": "DUMMY",
              "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_key": "A",
              "evidence_ids": ["F001"]}


def muat_gambar(path, max_side):
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    return img


def info_frame(p):
    n = int(re.search(r"(\d+)", p.stem).group(1))
    return {"id": f"F{n:03d}", "t_abs": (n - 1) * FRAME_INTERVAL + OFFSET_ABS}


def bangun_contoh(processor, messages, images, target_text):
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = processor(text=[prompt], images=images, return_tensors="pt")
    resp = processor.tokenizer(target_text + "<|im_end|>\n", add_special_tokens=False,
                               return_tensors="pt").input_ids
    contoh = dict(
        input_ids=torch.cat([p.input_ids, resp], 1)[0],
        labels=torch.cat([torch.full_like(p.input_ids, -100), resp], 1)[0],
        pixel_values=p.pixel_values, image_grid_thw=p.image_grid_thw)
    if "mm_token_type_ids" in p:
        contoh["mm_token_type_ids"] = torch.cat([p.mm_token_type_ids, torch.zeros_like(resp)], 1)[0]
    return prompt, contoh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1, help="ambil tiap frame ke-N")
    ap.add_argument("--max-side", type=int, default=None, help="sisi terpanjang untuk pemeriksaan penuh")
    ap.add_argument("--ukur", type=int, nargs="*", default=[0, 1024, 768, 512],
                    help="max_side yang diukur token visualnya (0 = asli)")
    ap.add_argument("--task", choices=["qg", "ae"], default="qg")
    a = ap.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    cfg = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    image_token_id = cfg["image_token_id"]
    merge = processor.image_processor.merge_size
    transkrip = baca_transkrip(TRANSKRIP, OFFSET_ABS)
    frames = FRAMES[::a.stride]
    print(f"frame: {len(frames)} dari {len(FRAMES)} | merge_size={merge}")

    # 1. token visual per frame
    print("\n== Token visual (frame_001 sebagai contoh) ==")
    for ms in a.ukur:
        img = muat_gambar(FRAMES[0], ms or None)
        out = processor.image_processor(images=[img], return_tensors="pt")
        tok = out["image_grid_thw"].prod(-1).item() // merge**2
        print(f"max_side={ms or 'asli'}: ukuran {img.size}, token/frame={tok}, "
              f"x{len(frames)} frame = {tok * len(frames)} token")

    # 2-4. satu contoh penuh
    print(f"\n== Contoh penuh task={a.task} (max_side={a.max_side or 'asli'}) ==")
    images = [muat_gambar(p, a.max_side) for p in frames]
    messages = susun_pesan(a.task, SEGMENT, [info_frame(p) for p in frames], transkrip,
                           soal=SOAL_DUMMY if a.task == "ae" else None)
    prompt, c = bangun_contoh(processor, messages, images, TARGET_DUMMY)

    n_img = (c["input_ids"] == image_token_id).sum().item()
    harap = c["image_grid_thw"].prod(-1).sum().item() // merge**2
    assert n_img == harap and n_img > 0, (n_img, harap)
    assert c["pixel_values"].numel() > 0
    n_lab = (c["labels"] != -100).sum().item()
    assert n_lab > 0
    decoded = processor.tokenizer.decode(c["input_ids"][c["labels"] != -100])
    assert decoded == TARGET_DUMMY + "<|im_end|>\n", repr(decoded)
    assert not (c["labels"][c["input_ids"] == image_token_id] != -100).any()
    full = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": TARGET_DUMMY}], tokenize=False)
    print("template latih == inferensi:", full == prompt + TARGET_DUMMY + "<|im_end|>\n")
    if full != prompt + TARGET_DUMMY + "<|im_end|>\n":
        print("  prefix cocok:", full.startswith(prompt))
        print("  ujung full  :", repr(full[len(prompt):][-80:]))
        print("  ujung rakit :", repr((prompt + TARGET_DUMMY + "<|im_end|>\n")[len(prompt):][-80:]))
    print(f"panjang total={len(c['input_ids'])} token | token gambar={n_img} | token label={n_lab}")
    print("pixel_values", tuple(c["pixel_values"].shape), "image_grid_thw", c["image_grid_thw"][:2].tolist(), "...")
    print("mm_token_type_ids ada:", "mm_token_type_ids" in c)
    print("label tersisa decode =", repr(decoded))
    print("\nOK: asersi 1-3 lolos (cek baris 'template latih == inferensi' di atas).")


if __name__ == "__main__":
    main()
