"""Uji 1 batch forward/backward QLoRA Qwen3.5-9B pada contoh pilot. JALANKAN DI COLAB (GPU).

Bukan training: tidak ada optimizer step. Tujuan: membuktikan sebelum training panjang bahwa
  - model dan LoRA terpasang pada modul yang benar (bukan visual, bukan mtp),
  - piksel gambar benar-benar sampai ke model, label hanya pada token assistant,
  - loss dan gradien finite, serta mengukur peak VRAM pada konfigurasi ini,
  - gambar berpengaruh pada loss (ablasi: frame asli vs frame kosong; pemeriksaan sanity, bukan evaluasi).

Model dimuat SEKALI, lalu beberapa konfigurasi (max_side:jumlah_frame) dicoba berurutan lewat --sweep,
supaya OOM tidak memaksa memuat ulang 19 GB.

Pakai (dari root repo; bobot di Drive/lokal Colab atau ID Hugging Face):
  python scripts/uji_1_batch.py --dry --model Qwen/Qwen3.5-9B
  python scripts/uji_1_batch.py --model <folder model> --sweep 1024:5,768:9
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cek_sampel as cs  # noqa: E402  (pakai muat_gambar & bangun_contoh yang sama)
from prompt_qg_ae import baca_transkrip, pilih_frame, susun_pesan  # noqa: E402

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


def bangun(processor, row, manifest, n_frame, max_side, potong):
    frames_m = pilih_frame(manifest["frames"], n_frame)
    seg = {"segment_id": manifest["segment_id"], "start": manifest["start_seconds"],
           "end": manifest["end_seconds"]}
    frames = [{"id": f["id"], "t_abs": f["t_abs"]} for f in frames_m]
    transkrip = baca_transkrip(PILOT / "transkrip.txt", cs.OFFSET_ABS)
    # evidence_ids di target harus merujuk frame/transkrip yang benar-benar ditampilkan
    target = json.loads(row["target"])
    bukti = set(target.get("evidence_ids") or [])
    for q in target.get("questions") or []:
        bukti |= set(q.get("evidence_ids") or [])
    hilang = bukti - ({f["id"] for f in frames} | {t["id"] for t in transkrip})
    assert not hilang, f"evidence_ids merujuk frame yang tidak ditampilkan: {sorted(hilang)}"
    messages = susun_pesan(row["task"], seg, frames, transkrip, soal=row["input_extra"])
    images = [cs.muat_gambar(ROOT / f["path"], max_side, potong) for f in frames_m]
    return messages, images, [f["id"] for f in frames]


def periksa_label_di_ujung(contoh):
    """Jawaban selalu di ujung urutan; syarat agar loss bisa dihitung hanya di posisi itu."""
    n = (contoh["labels"] != -100).sum().item()
    assert (contoh["labels"][:-n] == -100).all(), "label tidak hanya di ujung"
    assert (contoh["labels"][-n:] == contoh["input_ids"][-n:]).all(), "label != input_ids di ujung"
    return n


def ke_batch(contoh, device, dtype):
    b = {"input_ids": contoh["input_ids"][None], "labels": contoh["labels"][None],
         "attention_mask": torch.ones_like(contoh["input_ids"])[None],
         "pixel_values": contoh["pixel_values"].to(dtype),
         "image_grid_thw": contoh["image_grid_thw"]}
    if "mm_token_type_ids" in contoh:
        b["mm_token_type_ids"] = contoh["mm_token_type_ids"][None]
    return {k: v.to(device) for k, v in b.items()}


def hitung_loss(model, batch, n_label, hemat):
    """hemat=True: logits hanya untuk n_label+1 posisi terakhir (bukan seluruh ~7 rb posisi x 248 rb kosakata).
    Loss yang dihitung = cross-entropy token berikutnya pada token jawaban, sama dengan loss bawaan
    bila label hanya di ujung (diperiksa oleh periksa_label_di_ujung)."""
    if not hemat:
        return model(**batch).loss
    masuk = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**masuk, logits_to_keep=n_label + 1).logits[:, :-1]
    target = batch["input_ids"][:, -n_label:]
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))


def jalankan(model, processor, row, manifest, max_side, n_frame, potong, dtype, hemat):
    """Satu konfigurasi. Kembalikan dict hasil; OOM dicatat, bukan dilempar."""
    hasil = {"max_side": max_side, "n_frame_diminta": n_frame, "potong_pita": potong, "hemat_logits": hemat}
    fase = "bangun contoh"
    out = loss = grads = batch = None
    try:
        messages, images, frame_ids = bangun(processor, row, manifest, n_frame, max_side, potong)
        hasil["frame_ids"] = frame_ids
        prompt, contoh = cs.bangun_contoh(processor, messages, images, row["target"])
        n_label = periksa_label_di_ujung(contoh)
        merge = processor.image_processor.merge_size
        hasil.update(n_frame=len(images), total_token=len(contoh["input_ids"]), token_label=n_label,
                     token_gambar=contoh["image_grid_thw"].prod(-1).sum().item() // merge**2)
        n_img = (contoh["input_ids"] == model.config.image_token_id).sum().item()
        assert n_img == hasil["token_gambar"] and n_img > 0, (n_img, hasil["token_gambar"])
        print(f"\n== {max_side}px, {hasil['n_frame']} frame, potong_pita={potong}: {hasil['total_token']} token ==")

        batch = ke_batch(contoh, "cuda", dtype)
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fase = "maju (forward)"
        with torch.autocast("cuda", dtype=dtype):
            loss = hitung_loss(model, batch, n_label, hemat)
        assert torch.isfinite(loss), f"loss tidak finite: {loss}"
        fase = "mundur (backward)"
        loss.backward()
        torch.cuda.synchronize()
        hasil["detik_maju_mundur"] = time.time() - t0
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values()), "gradien None/NaN"
        hasil["loss"] = loss.item()
        hasil["vram_puncak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"loss {hasil['loss']:.4f} | puncak VRAM {hasil['vram_puncak_gib']:.2f} GiB | "
              f"{hasil['detik_maju_mundur']:.1f} dtk")
        model.zero_grad(set_to_none=True)
        loss = grads = None

        fase = "ablasi"
        model.eval()
        kosong = [Image.new("RGB", im.size, (128, 128, 128)) for im in images]
        _, contoh_k = cs.bangun_contoh(processor, messages, kosong, row["target"])
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            l_asli = hitung_loss(model, batch, n_label, hemat).item()
            l_kosong = hitung_loss(model, ke_batch(contoh_k, "cuda", dtype), n_label, hemat).item()
        hasil["loss_ablasi"] = {"frame_asli": l_asli, "frame_kosong": l_kosong}
        print(f"ablasi: loss frame asli {l_asli:.4f} | frame kosong {l_kosong:.4f}")
        hasil["hasil"] = "OK" if l_asli != l_kosong else "GAMBAR_TAMPAK_TIDAK_TERPAKAI"
    except torch.cuda.OutOfMemoryError:
        hasil["hasil"] = f"OOM pada fase: {fase}"
        hasil["vram_puncak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"OOM pada fase '{fase}' | puncak VRAM tercatat {hasil['vram_puncak_gib']:.2f} GiB")
    finally:
        out = loss = grads = batch = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "Model" / "Qwen3.5-9B"))
    ap.add_argument("--jsonl", default=str(PILOT / "contoh_latih.jsonl"))
    ap.add_argument("--contoh", type=int, default=0, help="indeks baris JSONL (0 = QG)")
    ap.add_argument("--sweep", default="1024:5,768:9",
                    help="daftar max_side:jumlah_frame dipisah koma, dicoba berurutan (frame merata, awal+akhir ikut)")
    ap.add_argument("--tanpa-potong-pita", action="store_true", help="jangan buang pita hitam (letterbox)")
    ap.add_argument("--logits-penuh", action="store_true",
                    help="pakai loss bawaan (logits seluruh posisi); defaultnya hemat VRAM")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--kbit-prep", action="store_true",
                    help="pakai prepare_model_for_kbit_training (upcast ke fp32; boros VRAM)")
    ap.add_argument("--sin-cudnn", action="store_true",
                    help="nonaktifkan cuDNN (jalan keluar bila cuDNN di runtime tidak serasi; konvolusi lebih lambat)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry", action="store_true", help="hanya data + token, tanpa memuat model")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    konfig = [tuple(int(x) for x in s.split(":")) for s in a.sweep.split(",")]

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    row = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")][a.contoh]
    manifest = json.loads((PILOT / "segmen.json").read_text(encoding="utf-8"))
    assert row["annotation_status"] == "diperiksa", "contoh belum berstatus diperiksa"
    info = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a), "paket": versi_paket(),
            "contoh_id": row["example_id"]}

    potong = not a.tanpa_potong_pita
    if a.dry:
        for ms, nf in konfig:
            messages, images, frame_ids = bangun(processor, row, manifest, nf, ms, potong)
            prompt, contoh = cs.bangun_contoh(processor, messages, images, row["target"])
            full = processor.apply_chat_template(
                messages + [{"role": "assistant", "content": row["target"]}], tokenize=False)
            assert full == prompt + row["target"] + "<|im_end|>\n", "template latih != inferensi"
            n = periksa_label_di_ujung(contoh)
            gambar = contoh["image_grid_thw"].prod(-1).sum().item() // processor.image_processor.merge_size**2
            print(f"{ms}px, {len(images)} frame, potong_pita={potong}: total {len(contoh['input_ids'])} token "
                  f"(gambar {gambar}, label {n}) | ukuran {images[1].size if len(images) > 1 else images[0].size} "
                  f"| {frame_ids[0]}..{frame_ids[-1]}")
        print("DRY: data dan token OK; model tidak dimuat.")
        return

    assert torch.cuda.is_available(), "butuh GPU (jalankan di Colab)"
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    # bf16 hanya bila GPU mendukung secara native (compute capability >= 8.0). T4 (7.5) hanya
    # "emulasi" bf16 yang lambat, walau torch.cuda.is_bf16_supported() melapor True.
    kapabilitas = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if kapabilitas[0] >= 8 else torch.float16
    if a.sin_cudnn:
        torch.backends.cudnn.enabled = False
    info.update(gpu=torch.cuda.get_device_name(0), kapabilitas=list(kapabilitas), dtype=str(dtype),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                cudnn_aktif=torch.backends.cudnn.enabled)
    print("GPU:", info["gpu"], kapabilitas, "| dtype:", dtype,
          f"| CUDA {info['cuda']} | cuDNN {info['cudnn']} (aktif={info['cudnn_aktif']})")
    torch.cuda.reset_peak_memory_stats()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, quantization_config=bnb, dtype=dtype, attn_implementation="sdpa", device_map={"": 0})
    model.config.use_cache = False
    info["detik_muat"] = time.time() - t0
    print("kelas model:", type(model).__name__)
    info["vram_setelah_muat_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(f"VRAM setelah muat: {info['vram_setelah_muat_gib']:.2f} GiB ({info['detik_muat']:.0f} dtk)")

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

    info["percobaan"] = []
    for ms, nf in konfig:
        h = jalankan(model, processor, row, manifest, ms, nf, potong, dtype, hemat=not a.logits_penuh)
        info["percobaan"].append(h)
        _simpan(info, a.out)
        if h["hasil"] == "OK":
            break  # konfigurasi terbesar yang muat sudah ketemu; sisanya lebih kecil
    print("\nRINGKASAN:")
    for h in info["percobaan"]:
        print(f"  {h['max_side']}px {h.get('n_frame', h['n_frame_diminta'])} frame: {h['hasil']}"
              + (f" | token {h.get('total_token')} | puncak {h.get('vram_puncak_gib', 0):.2f} GiB"))


def _simpan(info, out):
    path = Path(out) if out else ROOT / "runs" / f"uji_1_batch_{info['waktu'].replace(' ', '_').replace(':', '')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print("catatan disimpan:", path)


if __name__ == "__main__":
    main()
