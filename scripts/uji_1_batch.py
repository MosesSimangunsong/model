"""Uji 1 batch forward/backward QLoRA Qwen3.5-9B pada contoh pilot. JALANKAN DI COLAB (GPU).

Bukan training: tidak ada optimizer step. Tujuan: membuktikan sebelum training panjang bahwa
  - model dan LoRA terpasang pada modul yang benar (bukan visual, bukan mtp),
  - piksel gambar benar-benar sampai ke model, label hanya pada token assistant,
  - loss dan gradien finite, serta mengukur peak VRAM pada konfigurasi ini,
  - gambar berpengaruh pada loss (ablasi: frame asli vs frame kosong; pemeriksaan sanity, bukan evaluasi).

Pakai (dari root repo; bobot di Drive/lokal Colab atau ID Hugging Face):
  python scripts/uji_1_batch.py --dry                       # tanpa model, cek data + token (bisa di laptop)
  python scripts/uji_1_batch.py --model Qwen/Qwen3.5-9B --max-side 768 --stride 2
Kalau OOM, turunkan --max-side atau naikkan --stride, lalu catat konfigurasinya.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cek_sampel as cs  # noqa: E402  (pakai muat_gambar & bangun_contoh yang sama)
from prompt_qg_ae import baca_transkrip, susun_pesan  # noqa: E402

ROOT = cs.ROOT
PILOT = ROOT / "Data" / "Pilot"
TARGET = (r"model\.language_model\.layers\.\d+\."
          r"(self_attn\.(q|k|v|o)_proj"
          r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
          r"|mlp\.(gate|up|down)_proj)")


def versi_paket():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "flash-linear-attention", "causal-conv1d"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def bangun(processor, row, manifest, stride, max_side):
    frames_m = manifest["frames"][::stride]
    seg = {"segment_id": manifest["segment_id"], "start": manifest["start_seconds"],
           "end": manifest["end_seconds"]}
    frames = [{"id": f["id"], "t_abs": f["t_abs"]} for f in frames_m]
    transkrip = baca_transkrip(PILOT / "transkrip.txt", cs.OFFSET_ABS)
    messages = susun_pesan(row["task"], seg, frames, transkrip, soal=row["input_extra"])
    images = [cs.muat_gambar(ROOT / f["path"], max_side) for f in frames_m]
    return messages, images


def ke_batch(contoh, device, dtype):
    b = {"input_ids": contoh["input_ids"][None], "labels": contoh["labels"][None],
         "attention_mask": torch.ones_like(contoh["input_ids"])[None],
         "pixel_values": contoh["pixel_values"].to(dtype),
         "image_grid_thw": contoh["image_grid_thw"]}
    if "mm_token_type_ids" in contoh:
        b["mm_token_type_ids"] = contoh["mm_token_type_ids"][None]
    return {k: v.to(device) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "Model" / "Qwen3.5-9B"))
    ap.add_argument("--jsonl", default=str(PILOT / "contoh_latih.jsonl"))
    ap.add_argument("--contoh", type=int, default=0, help="indeks baris JSONL (0 = QG)")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1, help="ambil tiap frame ke-N")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--kbit-prep", action="store_true",
                    help="pakai prepare_model_for_kbit_training (upcast ke fp32; boros VRAM)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true", help="hanya data + token, tanpa memuat model")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    row = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")][a.contoh]
    manifest = json.loads((PILOT / "segmen.json").read_text(encoding="utf-8"))
    messages, images = bangun(processor, row, manifest, a.stride, a.max_side)
    prompt, contoh = cs.bangun_contoh(processor, messages, images, row["target"])
    full = processor.apply_chat_template(
        messages + [{"role": "assistant", "content": row["target"]}], tokenize=False)
    assert full == prompt + row["target"] + "<|im_end|>\n", "template latih != inferensi"
    assert row["annotation_status"] == "diperiksa", "contoh belum berstatus diperiksa"
    merge = processor.image_processor.merge_size
    info = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "paket": versi_paket(),
            "contoh_id": row["example_id"], "n_frame": len(images),
            "total_token": len(contoh["input_ids"]),
            "token_gambar": contoh["image_grid_thw"].prod(-1).sum().item() // merge**2,
            "token_label": (contoh["labels"] != -100).sum().item()}
    print(json.dumps({k: v for k, v in info.items() if k != "args"}, indent=1, ensure_ascii=False))
    assert (contoh["labels"] != -100).sum() > 0 and contoh["pixel_values"].numel() > 0
    if a.dry:
        print("DRY: data dan token OK; model tidak dimuat.")
        return

    assert torch.cuda.is_available(), "butuh GPU (jalankan di Colab)"
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    info["gpu"] = torch.cuda.get_device_name(0)
    info["dtype"] = str(dtype)
    print("GPU:", info["gpu"], "| dtype:", dtype)
    torch.cuda.reset_peak_memory_stats()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, quantization_config=bnb, dtype=dtype, attn_implementation="sdpa", device_map={"": 0})
    model.config.use_cache = False
    print("kelas model:", type(model).__name__)
    info["vram_setelah_muat_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"VRAM setelah muat: {info['vram_setelah_muat_gib']:.2f} GiB")

    if a.kbit_prep:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                             target_modules=TARGET, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    train = [n for n, p in model.named_parameters() if p.requires_grad]
    assert train and all("lora_" in n for n in train), train[:10]
    assert not any("visual" in n for n in train) and not any(".mtp." in n for n in train)
    lapisan = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in train if ".layers." in n})
    print("lapisan terkena LoRA:", lapisan[0], "..", lapisan[-1], f"({len(lapisan)} lapisan)")
    info["lapisan_lora"] = len(lapisan)

    cfg = model.config
    n_img = (contoh["input_ids"] == cfg.image_token_id).sum().item()
    assert n_img == info["token_gambar"] and n_img > 0, (n_img, info["token_gambar"])

    batch = ke_batch(contoh, "cuda", dtype)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        with torch.autocast("cuda", dtype=dtype):
            out = model(**batch)
        assert torch.isfinite(out.loss), out.loss
        out.loss.backward()
        torch.cuda.synchronize()
        info["detik_maju_mundur"] = time.time() - t0
    except torch.cuda.OutOfMemoryError:
        info["hasil"] = "OOM"
        info["vram_puncak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"OOM pada max_side={a.max_side}, stride={a.stride}, total_token={info['total_token']}")
        _simpan(info, a.out)
        raise SystemExit(2)
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values()), "gradien None/NaN"
    info["loss"] = out.loss.item()
    info["vram_puncak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"loss {info['loss']:.4f} | puncak VRAM {info['vram_puncak_gib']:.2f} GiB | "
          f"{info['detik_maju_mundur']:.1f} dtk")
    model.zero_grad(set_to_none=True)
    del out, grads

    # ablasi: frame asli vs frame kosong (ukuran sama -> jumlah token sama)
    model.eval()
    kosong = [Image.new("RGB", im.size, (128, 128, 128)) for im in images]
    _, contoh_k = cs.bangun_contoh(processor, messages, kosong, row["target"])
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        l_asli = model(**ke_batch(contoh, "cuda", dtype)).loss.item()
        l_kosong = model(**ke_batch(contoh_k, "cuda", dtype)).loss.item()
    info["loss_ablasi"] = {"frame_asli": l_asli, "frame_kosong": l_kosong}
    print(f"ablasi: loss frame asli {l_asli:.4f} | frame kosong {l_kosong:.4f}")
    assert l_asli != l_kosong, "loss identik: gambar tampaknya tidak terpakai"
    info["hasil"] = "OK"
    _simpan(info, a.out)


def _simpan(info, out):
    path = Path(out) if out else ROOT / "runs" / f"uji_1_batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print("catatan disimpan:", path)


if __name__ == "__main__":
    main()
