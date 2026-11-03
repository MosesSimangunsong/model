"""Mini-training QLoRA pilot Qwen3.5-9B (QG + AE). Spesifikasi: Instruksi_Claude_Mini_Training_Qwen35_LeveLearn.md.

Alur mode train (satu proses, model dimuat sekali):
  A preflight data/token -> B generasi QG model dasar -> C pasang LoRA + snapshot -> D loss awal 4 contoh
  -> E mini-training (AMP fp16 + GradScaler) -> F loss akhir, simpan adapter, generasi QG sesudah latih.
Mode reload (proses baru): muat model dasar + adapter tersimpan, bandingkan state, loss, dan generasi.

Hasil hanya membuktikan mekanik alur (4 contoh dari 1 sumber), bukan kualitas atau generalisasi.

Pakai (dari root repo):
  python scripts/latih_pilot.py --mode train --dry --model <folder model> --output-dir <folder run>
  python scripts/latih_pilot.py --mode train --model <folder model> --output-dir <folder run>
  python scripts/latih_pilot.py --mode reload --run-dir <folder run>
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cek_sampel as cs  # noqa: E402
import uji_1_batch as u  # noqa: E402  (bangun, ke_batch, hitung_loss, periksa_label_di_ujung, TARGET)
from prompt_qg_ae import baca_transkrip, pilih_frame, validasi_ae, validasi_qg  # noqa: E402

ROOT = cs.ROOT
PILOT = ROOT / "Data" / "Pilot"
IM_END = "<|im_end|>"


# ---------- utilitas catatan ----------
def tulis_json(path, obj):
    """Tulis atomik: berkas sementara lalu ganti, agar log tidak setengah jadi bila sesi putus."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


class Fase:
    """Catat waktu dan puncak memori GPU per fase (peak di-reset tiap fase)."""

    def __init__(self):
        self.catatan = {}

    def mulai(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()

    def selesai(self, nama, **extra):
        rec = {"detik": None, **extra}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            rec["puncak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            rec["puncak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        rec["detik"] = time.time() - self.t0
        self.catatan[nama] = rec
        return rec


# ---------- fungsi murni (diuji unit di laptop) ----------
def status_langkah(norm, scale_lama, scale_baru, scaler_aktif):
    """Update dianggap berhasil hanya bila norm gradien finite dan GradScaler tidak menurunkan scale.
    Return value optimizer.step() (biasanya None) tidak dipakai sebagai detektor."""
    if not torch.isfinite(torch.as_tensor(norm)):
        return "dilewati"
    if scaler_aktif and scale_baru < scale_lama:
        return "dilewati"
    return "update"


def parse_ketat(raw):
    """Parsing JSON ketat tanpa perbaikan. Catat pagar Markdown bila ada, tapi tidak dibuang."""
    hasil = {"parse_status": None, "objek": None, "galat": None,
             "berpagar_markdown": raw.lstrip().startswith("```")}
    try:
        hasil["objek"] = json.loads(raw.strip())
        hasil["parse_status"] = "terbaca"
    except json.JSONDecodeError as e:
        hasil["parse_status"] = "tidak_terbaca"
        hasil["galat"] = f"{e.msg} (baris {e.lineno}, kolom {e.colno})"
    return hasil


def periksa_pemetaan_ae(rows):
    """Setiap contoh AE harus memakai soal yang sama persis dengan soal pada target QG."""
    qg = [r for r in rows if r["task"] == "qg"]
    galat = []
    if len(qg) != 1:
        return [f"harus tepat 1 contoh QG, ada {len(qg)}"]
    soal = {q["id"]: q for q in json.loads(qg[0]["target"])["questions"]}
    for r in rows:
        if r["task"] != "ae":
            continue
        s, t = r["input_extra"] or {}, json.loads(r["target"])
        if s.get("id") not in soal:
            galat.append(f"{r['example_id']}: soal {s.get('id')} tidak ada di target QG")
        elif s != soal[s["id"]]:
            galat.append(f"{r['example_id']}: input_extra berbeda dari soal {s['id']} di target QG")
        if t.get("question_id") != s.get("id"):
            galat.append(f"{r['example_id']}: question_id target {t.get('question_id')} != soal {s.get('id')}")
    return galat


def loss_berbobot(per_contoh, task=None):
    rs = [r for r in per_contoh if task is None or r["task"] == task]
    n = sum(r["token_target"] for r in rs)
    return sum(r["loss"] * r["token_target"] for r in rs) / n if n else None


# ---------- data ----------
def muat_data(a):
    if not PILOT.is_dir():
        kecil = ROOT / "data" / "pilot"
        raise SystemExit(f"Folder {PILOT} tidak ada" + (f"; yang ada {kecil} (huruf kecil). Ganti nama ke Data/Pilot."
                                                          if kecil.is_dir() else "."))
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")]
    manifest = json.loads((PILOT / "segmen.json").read_text(encoding="utf-8"))
    return rows, manifest


def preflight(processor, rows, manifest, a):
    """Tahap A. Kesalahan fatal -> SystemExit dengan example_id; peringatan dicatat."""
    fatal, peringatan, per_contoh = [], [], []
    for r in rows:
        if r["annotation_status"] != "diperiksa":
            fatal.append(f"{r['example_id']}: annotation_status={r['annotation_status']}")
    fatal += periksa_pemetaan_ae(rows)
    durasi = manifest["end_seconds"] - manifest["start_seconds"]
    for t in manifest["transcript"]["segments"]:
        if not (0 <= t["t_rel_start"] < t["t_rel_end"] <= durasi):
            peringatan.append(f"{t['id']}: waktu relatif {t['t_rel_start']}-{t['t_rel_end']} melewati durasi "
                              f"segmen {durasi} dtk (berasal dari keluaran Whisper; tidak diubah)")
    merge = processor.image_processor.merge_size
    for r in rows:
        try:
            messages, images, frame_ids = u.bangun(processor, r, manifest, a.n_frame, a.max_side, a.potong)
        except AssertionError as e:
            fatal.append(f"{r['example_id']}: {e}")
            continue
        prompt, c = cs.bangun_contoh(processor, messages, images, r["target"])
        n = u.periksa_label_di_ujung(c)
        decoded = processor.tokenizer.decode(c["input_ids"][-n:])
        full = processor.apply_chat_template(messages + [{"role": "assistant", "content": r["target"]}],
                                             tokenize=False)
        cek = {
            "label_sama_target": decoded == r["target"] + IM_END + "\n",
            "template_latih_sama_inferensi": full == prompt + r["target"] + IM_END + "\n",
            "target_tidak_di_prompt": r["target"] not in prompt,
        }
        for k, v in cek.items():
            if not v:
                fatal.append(f"{r['example_id']}: {k} gagal")
        tok_img = c["image_grid_thw"].prod(-1).sum().item() // merge**2
        ukuran = []
        for fid in frame_ids:
            f = next(x for x in manifest["frames"] if x["id"] == fid)
            asli = cs.Image.open(ROOT / f["path"]).size
            ukuran.append({"id": fid, "asli": asli, "dipakai": cs.muat_gambar(ROOT / f["path"], a.max_side, a.potong).size})
        per_contoh.append({"example_id": r["example_id"], "task": r["task"], "frame_ids": frame_ids,
                           "ukuran_frame": ukuran, "token_total": len(c["input_ids"]), "token_gambar": tok_img,
                           "token_target": n, "token_prompt": len(c["input_ids"]) - n, **cek})
    lap = {"fatal": fatal, "peringatan": peringatan, "contoh": per_contoh,
           "catatan": "status 'diperiksa' adalah metadata anotasi; kode tidak memverifikasi ulang isinya"}
    return lap


def data_manifest(rows, manifest, frame_ids, a):
    berkas = [Path(a.jsonl), PILOT / "segmen.json", PILOT / "transkrip.txt"]
    berkas += [ROOT / f["path"] for f in manifest["frames"] if f["id"] in frame_ids]
    skrip = [Path(__file__).parent / n for n in ("latih_pilot.py", "uji_1_batch.py", "cek_sampel.py", "prompt_qg_ae.py")]
    return {"source_group_id": sorted({r["source_group_id"] for r in rows}),
            "urutan_contoh": [r["example_id"] for r in rows],
            "hash_data": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p) for p in berkas},
            "hash_skrip": {p.name: sha256(p) for p in skrip}}


# ---------- model ----------
def muat_model(model_path, dtype):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, quantization_config=bnb, dtype=dtype, attn_implementation="sdpa", device_map={"": 0})
    model.config.use_cache = False
    return model


def lingkungan(dtype):
    info = {"python": platform.python_version(), "paket": u.versi_paket(), "dtype": str(dtype),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(0), kapabilitas=list(torch.cuda.get_device_capability(0)),
                    total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cudnn_aktif=torch.backends.cudnn.enabled)
    return info


def pilih_dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def generasi(model, processor, row, manifest, a, dtype, out_dir, prefix):
    """Generasi tanpa target: hanya prompt + gambar. Simpan raw text dan token ID sebelum parsing."""
    hasil = {"generation_status": "tidak_dijalankan", "example_id": row["example_id"]}
    try:
        messages, images, frame_ids = u.bangun(processor, row, manifest, a.n_frame, a.max_side, a.potong)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        assert row["target"] not in prompt, "target bocor ke prompt generasi"
        masuk = processor(text=[prompt], images=images, return_tensors="pt")
        masuk = {k: (v.to(dtype) if k == "pixel_values" else v).to("cuda") for k, v in masuk.items()}
        tok = processor.tokenizer
        eos = [tok.convert_tokens_to_ids(IM_END)]
        model.eval()
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            out = model.generate(**masuk, max_new_tokens=a.max_new_tokens, do_sample=False, num_beams=1,
                                 num_return_sequences=1, use_cache=True, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        baru = out[0, masuk["input_ids"].shape[1]:].tolist()
        del out, masuk
        raw = tok.decode(baru, skip_special_tokens=False)
        teks = raw[: -len(IM_END)] if raw.endswith(IM_END) else raw  # penanda akhir bukan bagian JSON
        berhenti = "eos" if baru and baru[-1] in eos else ("batas_max_new_tokens" if len(baru) >= a.max_new_tokens
                                                           else "lain")
        (out_dir / f"{prefix}.raw.txt").write_text(raw, encoding="utf-8")
        tulis_json(out_dir / f"{prefix}.tokens.json", {
            "token_baru": baru, "jumlah_token_baru": len(baru), "alasan_berhenti": berhenti,
            "max_new_tokens": a.max_new_tokens, "detik": time.time() - t0,
            "decoding": {"do_sample": False, "num_beams": 1, "eos_token_id": eos, "enable_thinking": False},
            "frame_ids": frame_ids})
        p = parse_ketat(teks)
        tulis_json(out_dir / f"{prefix}.parsed.json", p)
        id_disajikan = set(frame_ids) | {t["id"] for t in baca_transkrip(PILOT / "transkrip.txt", cs.OFFSET_ABS)}
        val = {"schema_status": "tidak_diperiksa", "galat": None, "task_status": None}
        if p["objek"] is not None and isinstance(p["objek"], dict):
            galat = (validasi_qg(p["objek"], id_disajikan) if row["task"] == "qg"
                     else validasi_ae(p["objek"], row["input_extra"], id_disajikan))
            val = {"schema_status": "sesuai" if not galat else "tidak_sesuai", "galat": galat,
                   "task_status": p["objek"].get("status")}
        val["semantic_review"] = "belum_diperiksa"
        tulis_json(out_dir / f"{prefix}.validation.json", val)
        hasil.update(generation_status="berhasil", alasan_berhenti=berhenti, jumlah_token_baru=len(baru),
                     detik=time.time() - t0, parse_status=p["parse_status"], **val)
    except torch.cuda.OutOfMemoryError:
        hasil["generation_status"] = "OOM"
    except Exception as e:  # dicatat apa adanya, tidak disembunyikan
        hasil.update(generation_status="runtime_error", galat=f"{type(e).__name__}: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return hasil


def loss_semua(model, processor, rows, manifest, a, dtype):
    """Loss evaluasi (eval, tanpa gradien) per contoh dengan prosedur tetap."""
    model.eval()
    hasil = []
    for r in rows:
        messages, images, _ = u.bangun(processor, r, manifest, a.n_frame, a.max_side, a.potong)
        _, c = cs.bangun_contoh(processor, messages, images, r["target"])
        n = u.periksa_label_di_ujung(c)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            loss = u.hitung_loss(model, u.ke_batch(c, "cuda", dtype), n, hemat=True).item()
        hasil.append({"example_id": r["example_id"], "task": r["task"], "loss": loss, "token_target": n})
        torch.cuda.empty_cache()
    return hasil


def ringkas_loss(per_contoh):
    return {"per_contoh": per_contoh, "berbobot_token_semua": loss_berbobot(per_contoh),
            "berbobot_token_qg": loss_berbobot(per_contoh, "qg"), "berbobot_token_ae": loss_berbobot(per_contoh, "ae"),
            "rumus": "sum(loss_i * token_target_i) / sum(token_target_i)",
            "catatan": "loss pada data latih pilot, bukan validation/test loss"}


def cek_ekuivalensi(model, processor, rows, manifest, dtype):
    """Loss hemat (logits_to_keep) vs loss bawaan pada input kecil: contoh AE terpendek, 2 frame, 512 px."""
    r = min((x for x in rows if x["task"] == "ae"), key=lambda x: len(x["target"]))
    messages, images, _ = u.bangun(processor, r, manifest, 2, 512, True)
    _, c = cs.bangun_contoh(processor, messages, images, r["target"])
    n = u.periksa_label_di_ujung(c)
    b = u.ke_batch(c, "cuda", dtype)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        hemat = u.hitung_loss(model, b, n, hemat=True).item()
        torch.cuda.empty_cache()
        standar = u.hitung_loss(model, b, n, hemat=False).item()
    torch.cuda.empty_cache()
    return {"example_id": r["example_id"], "token_total": len(c["input_ids"]), "loss_hemat": hemat,
            "loss_standar": standar, "selisih_abs": abs(hemat - standar), "toleransi": 1e-3,
            "setara": abs(hemat - standar) <= 1e-3}


def state_adapter(model):
    from peft import get_peft_model_state_dict
    return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def bandingkan_state(awal, akhir):
    dlt = {k: (akhir[k] - awal[k]) for k in awal}
    berubah = [k for k, d in dlt.items() if d.abs().max().item() > 0]
    def ringkas(filter_):
        ks = [k for k in dlt if filter_(k)]
        return {"jumlah_tensor": len(ks), "berubah": sum(k in berubah for k in ks),
                "max_abs_delta": max((dlt[k].abs().max().item() for k in ks), default=0.0),
                "norm_delta": sum(dlt[k].float().pow(2).sum().item() for k in ks) ** 0.5}
    return {"semua": ringkas(lambda k: True), "lora_A": ringkas(lambda k: "lora_A" in k),
            "lora_B": ringkas(lambda k: "lora_B" in k),
            "semua_finite": all(torch.isfinite(v).all().item() for v in akhir.values())}


# ---------- mode train ----------
def mode_train(a):
    from transformers import AutoProcessor, set_seed
    set_seed(a.seed)
    out = Path(a.output_dir)
    if (out / "adapter").exists() or (out / "train_steps.jsonl").exists():
        raise SystemExit(f"{out} sudah berisi hasil training; pakai folder run baru")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"tahap": {}, "mulai": time.strftime("%Y-%m-%d %H:%M:%S")}
    processor = AutoProcessor.from_pretrained(a.model)
    rows, manifest = muat_data(a)

    # A. preflight
    lap = preflight(processor, rows, manifest, a)
    tulis_json(out / "preflight.json", lap)
    frame_ids = lap["contoh"][0]["frame_ids"] if lap["contoh"] else []
    tulis_json(out / "data_manifest.json", {**data_manifest(rows, manifest, frame_ids, a),
                                            "token": [{k: c[k] for k in ("example_id", "token_total", "token_gambar",
                                                                         "token_target", "token_prompt")}
                                                      for c in lap["contoh"]]})
    for c in lap["contoh"]:
        print(f"{c['example_id']}: total {c['token_total']} | gambar {c['token_gambar']} | target {c['token_target']}")
    for w in lap["peringatan"]:
        print("PERINGATAN:", w)
    if lap["fatal"]:
        raise SystemExit("Preflight gagal:\n- " + "\n- ".join(lap["fatal"]))
    summary["tahap"]["A_preflight"] = "lolos" + (" dengan peringatan" if lap["peringatan"] else "")
    konfig = {**vars(a), "frame_ids": frame_ids, "optimizer": {"nama": "AdamW", "lr": a.lr, "betas": [0.9, 0.999],
              "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False},
              "max_grad_norm": 1.0, "scheduler": "konstan, tanpa warmup", "batch": 1, "grad_accumulation": 1,
              "lora": {"r": a.r, "alpha": a.alpha, "dropout": a.dropout, "target_modules": u.TARGET,
                       "jenis": "LoRA standar (B diinisialisasi nol)"},
              "kuantisasi": "4-bit NF4, double quant", "attn_implementation": "sdpa",
              "gradient_checkpointing": {"use_reentrant": False}}
    if a.dry:
        tulis_json(out / "config_resolved.json", {**konfig, "catatan": "dry: model tidak dimuat"})
        tulis_json(out / "summary.json", summary)
        print("DRY: preflight lolos; model tidak dimuat. Catatan di", out)
        return

    assert torch.cuda.is_available(), "butuh GPU"
    if a.sin_cudnn:
        torch.backends.cudnn.enabled = False
    dtype = pilih_dtype()
    konfig["dtype"] = str(dtype)
    konfig["model_config_sha256"] = sha256(Path(a.model) / "config.json")
    tulis_json(out / "environment.json", lingkungan(dtype))
    tulis_json(out / "config_resolved.json", konfig)
    fase = Fase()

    fase.mulai()
    model = muat_model(a.model, dtype)
    fase.selesai("muat_model", vram_setelah_muat_gib=torch.cuda.memory_allocated() / 2**30)
    qg_row = next(r for r in rows if r["task"] == "qg")

    # B. generasi model dasar
    fase.mulai()
    base = generasi(model, processor, qg_row, manifest, a, dtype, out, "baseline_qg")
    fase.selesai("generasi_baseline_qg")
    summary["tahap"]["B_baseline_qg"] = base
    tulis_json(out / "summary.json", summary)
    print("baseline QG:", {k: base.get(k) for k in ("generation_status", "alasan_berhenti", "jumlah_token_baru",
                                                    "parse_status", "schema_status")})
    if base["generation_status"] != "berhasil" and not a.lanjut_tanpa_baseline:
        raise SystemExit("Generasi baseline gagal; jalankan ulang atau pakai --lanjut-tanpa-baseline (hasil tidak lengkap)")
    if a.gen_ae:
        summary["tahap"]["B_baseline_ae"] = [generasi(model, processor, r, manifest, a, dtype, out,
                                                      f"baseline_ae_{r['input_extra']['id']}")
                                             for r in rows if r["task"] == "ae"]
    else:
        summary["tahap"]["B_baseline_ae"] = "ae_generation_pending"

    # C. pasang LoRA
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                                             target_modules=u.TARGET, task_type="CAUSAL_LM"))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, _ in params), [n for n, _ in params][:5]
    assert not any("visual" in n or ".mtp." in n or "embed_tokens" in n or "lm_head" in n for n, _ in params)
    dicast = [n for n, p in params if p.dtype != torch.float32]
    for _, p in params:  # GradScaler butuh gradien fp32 pada parameter yang dioptimasi
        if p.dtype != torch.float32:
            p.data = p.data.float()
    lapisan = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, _ in params})
    tulis_json(out / "trainable_parameters.json", {
        "jumlah_tensor": len(params), "jumlah_parameter": sum(p.numel() for _, p in params),
        "jumlah_parameter_total": sum(p.numel() for p in model.parameters()),
        "dtype": sorted({str(p.dtype) for _, p in params}), "dicast_ke_fp32": len(dicast),
        "lapisan": lapisan, "nama": [n for n, _ in params],
        "dibekukan": "visual, merger, embed_tokens, lm_head, mtp, bobot dasar (vision encoder tetap forward)"})
    awal = state_adapter(model)
    summary["tahap"]["C_adapter"] = {"tensor": len(params), "lapisan": len(lapisan),
                                     "lora_B_awal_nol": all(v.abs().max().item() == 0 for k, v in awal.items()
                                                            if "lora_B" in k)}
    print("adapter:", summary["tahap"]["C_adapter"])

    # D. ekuivalensi loss + loss awal
    try:
        ekv = cek_ekuivalensi(model, processor, rows, manifest, dtype)
    except torch.cuda.OutOfMemoryError:
        ekv = {"status": "OOM", "catatan": "belum terverifikasi"}
        torch.cuda.empty_cache()
    summary["tahap"]["D_ekuivalensi_loss"] = ekv
    print("ekuivalensi loss hemat vs standar:", ekv)
    fase.mulai()
    loss_awal = ringkas_loss(loss_semua(model, processor, rows, manifest, a, dtype))
    fase.selesai("loss_awal")
    print("loss awal (berbobot):", loss_awal["berbobot_token_semua"])

    # E. mini-training
    opt = torch.optim.AdamW([p for _, p in params], lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                            foreach=False, fused=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    berhasil, percobaan, status_e = 0, 0, "berjalan"
    log_langkah = open(out / "train_steps.jsonl", "a", encoding="utf-8")
    try:
        while berhasil < a.steps:
            if percobaan >= a.maks_percobaan:
                status_e = "berhenti_batas_percobaan"
                break
            r = rows[percobaan % len(rows)]
            percobaan += 1
            messages, images, _ = u.bangun(processor, r, manifest, a.n_frame, a.max_side, a.potong)
            _, c = cs.bangun_contoh(processor, messages, images, r["target"])
            n = u.periksa_label_di_ujung(c)
            batch = u.ke_batch(c, "cuda", dtype)
            model.train()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                loss = u.hitung_loss(model, batch, n, hemat=True)
            if not torch.isfinite(loss):
                status_e = f"loss_tidak_finite pada {r['example_id']}"
                break
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            kosong = sum(p.grad is None for _, p in params)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
            scale_lama = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(opt)
            scaler.update()
            scale_baru = scaler.get_scale() if scaler.is_enabled() else 1.0
            st = status_langkah(norm, scale_lama, scale_baru, scaler.is_enabled())
            berhasil += st == "update"
            torch.cuda.synchronize()
            rec = {"percobaan": percobaan, "example_id": r["example_id"], "task": r["task"], "loss": loss.item(),
                   "lr": a.lr, "scale_sebelum": scale_lama, "scale_sesudah": scale_baru,
                   "grad_norm_sebelum_clip": norm.item(), "gradien_kosong": kosong, "status": st,
                   "update_berhasil_kumulatif": berhasil, "detik": time.time() - t0,
                   "puncak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "puncak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
            log_langkah.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_langkah.flush()
            print(f"[{percobaan}] {r['example_id']}: loss {rec['loss']:.4f} | {st} | norm {rec['grad_norm_sebelum_clip']:.3g}"
                  f" | {rec['detik']:.1f} dtk | puncak {rec['puncak_allocated_gib']:.2f} GiB")
            del loss, batch
            if kosong:
                status_e = f"gradien_kosong ({kosong} tensor) pada {r['example_id']}"
                break
        else:
            status_e = "selesai"
    except torch.cuda.OutOfMemoryError:
        status_e = f"OOM pada percobaan {percobaan}"
    finally:
        log_langkah.close()
    akhir_state = state_adapter(model)
    cek_update = {**bandingkan_state(awal, akhir_state), "update_berhasil": berhasil, "target_update": a.steps,
                  "percobaan": percobaan, "target_tercapai": berhasil == a.steps, "status": status_e,
                  "rasio_kunjungan": "1 QG : 3 AE per putaran (objective tidak otomatis seimbang)"}
    tulis_json(out / "adapter_update_check.json", cek_update)
    summary["tahap"]["E_training"] = {k: cek_update[k] for k in ("status", "update_berhasil", "percobaan",
                                                                 "target_tercapai", "semua_finite")}
    tulis_json(out / "summary.json", summary)
    print("training:", summary["tahap"]["E_training"], "| tensor berubah:", cek_update["semua"]["berubah"])
    if status_e != "selesai" or berhasil == 0:
        raise SystemExit(f"Training tidak selesai: {status_e}. Lihat train_steps.jsonl")

    # F. loss akhir, simpan, generasi sesudah latih
    del opt, scaler
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    loss_akhir = ringkas_loss(loss_semua(model, processor, rows, manifest, a, dtype))
    tulis_json(out / "loss_before_after.json", {"awal": loss_awal, "akhir": loss_akhir,
                                                "catatan": "turunnya loss latih adalah pemeriksaan kewajaran, bukan bukti kualitas"})
    print("loss akhir (berbobot):", loss_akhir["berbobot_token_semua"])
    adir = out / "adapter"
    model.save_pretrained(adir, safe_serialization=True)
    processor.save_pretrained(out / "processor")
    from safetensors.torch import load_file
    disimpan = load_file(adir / "adapter_model.safetensors")
    kunci_sama = set(disimpan) == set(akhir_state)
    selisih = max((disimpan[k].float() - akhir_state[k]).abs().max().item() for k in disimpan) if kunci_sama else None
    summary["tahap"]["F_simpan"] = {"adapter_dir": str(adir), "kunci_sama": kunci_sama, "selisih_maks": selisih,
                                    "finite": all(torch.isfinite(v).all().item() for v in disimpan.values()),
                                    "reload": "reload_pending"}
    tulis_json(out / "reload_check.json", {"status": "reload_pending"})
    tulis_json(out / "summary.json", summary)
    fase.mulai()
    after = generasi(model, processor, qg_row, manifest, a, dtype, out, "after_qg")
    fase.selesai("generasi_after_qg")
    summary["tahap"]["F_after_qg"] = after
    summary["tahap"]["F_after_ae"] = ([generasi(model, processor, r, manifest, a, dtype, out,
                                                f"after_ae_{r['input_extra']['id']}") for r in rows if r["task"] == "ae"]
                                      if a.gen_ae else "ae_generation_pending")
    summary["fase"] = fase.catatan
    summary["selesai"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tulis_json(out / "summary.json", summary)
    tulis_review(out, base, after)
    print("after QG:", {k: after.get(k) for k in ("generation_status", "alasan_berhenti", "jumlah_token_baru",
                                                  "parse_status", "schema_status")})
    print("Selesai. Lanjutkan dengan --mode reload --run-dir", out)


def tulis_review(out, base, after):
    baris = ["# Lembar pemeriksaan manual QG (isi oleh pemeriksa)", "",
             "Status otomatis tidak membuktikan soal benar. Periksa isi, kunci, pengecoh, dan notasi.", "",
             "| Keluaran | generasi | parsing | skema | status tugas | Sesuai sumber | Dapat dijawab | 1 kunci benar "
             "| Pengecoh masuk akal | Notasi tepat | Catatan (operator/negasi/kurung/nilai/inferensi) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nama, h in (("baseline_qg", base), ("after_qg", after)):
        baris.append(f"| {nama} | {h.get('generation_status')} | {h.get('parse_status')} | {h.get('schema_status')} "
                     f"| {h.get('task_status')} |  |  |  |  |  |  |")
    (out / "review_template.md").write_text("\n".join(baris) + "\n", encoding="utf-8")


# ---------- mode reload ----------
def mode_reload(a):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoProcessor, set_seed
    run = Path(a.run_dir)
    kf = json.loads((run / "config_resolved.json").read_text(encoding="utf-8"))
    for k in ("model", "jsonl", "max_side", "n_frame", "potong", "max_new_tokens", "seed"):
        setattr(a, k, kf[k])
    set_seed(a.seed)
    assert sha256(Path(a.model) / "config.json") == kf["model_config_sha256"], "model dasar berbeda dari run"
    if kf.get("sin_cudnn"):
        torch.backends.cudnn.enabled = False
    dtype = pilih_dtype()
    assert str(dtype) == kf["dtype"], f"dtype {dtype} != {kf['dtype']} saat training"
    processor = AutoProcessor.from_pretrained(a.model)
    rows, manifest = muat_data(a)
    hasil = {"status": "berjalan", "proses_baru": True, "lingkungan": lingkungan(dtype)}
    model = muat_model(a.model, dtype)
    model = PeftModel.from_pretrained(model, run / "adapter", is_trainable=False)
    termuat = state_adapter(model)
    disimpan = load_file(run / "adapter" / "adapter_model.safetensors")
    hasil["state"] = {"kunci_sama": set(termuat) == set(disimpan),
                      "selisih_maks": max((termuat[k] - disimpan[k].float()).abs().max().item() for k in disimpan)
                      if set(termuat) == set(disimpan) else None,
                      "lora_B_nonnol": any(v.abs().max().item() > 0 for k, v in termuat.items() if "lora_B" in k)}
    loss_reload = ringkas_loss(loss_semua(model, processor, rows, manifest, a, dtype))
    lb = json.loads((run / "loss_before_after.json").read_text(encoding="utf-8"))["akhir"]
    hasil["loss"] = {"reload": loss_reload, "in_memory_akhir": lb["berbobot_token_semua"],
                     "selisih_berbobot": abs(loss_reload["berbobot_token_semua"] - lb["berbobot_token_semua"])}
    qg_row = next(r for r in rows if r["task"] == "qg")
    g = generasi(model, processor, qg_row, manifest, a, dtype, run, "reload_qg")
    sebelum = (run / "after_qg.raw.txt")
    if g["generation_status"] == "berhasil" and sebelum.exists():
        x, y = sebelum.read_text(encoding="utf-8"), (run / "reload_qg.raw.txt").read_text(encoding="utf-8")
        beda = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y)))
        g["sama_dengan_after_qg"] = x == y
        g["indeks_beda_pertama"] = beda
    hasil["generasi_qg"] = g
    hasil["status"] = "selesai"
    hasil["catatan"] = "greedy tidak menjamin identik lintas backend; perbedaan dicatat, tidak ditutupi"
    tulis_json(run / "reload_check.json", hasil)
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    s["tahap"]["G_reload"] = {"status": "selesai", "state": hasil["state"], "selisih_loss": hasil["loss"]["selisih_berbobot"],
                              "generasi": g.get("generation_status"), "sama_dengan_after": g.get("sama_dengan_after_qg")}
    tulis_json(run / "summary.json", s)
    print(json.dumps(s["tahap"]["G_reload"], indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["train", "reload"], required=True)
    ap.add_argument("--dry", action="store_true", help="train: preflight data/token saja, tanpa memuat model")
    ap.add_argument("--model", default=str(ROOT / "Model" / "Qwen3.5-9B"))
    ap.add_argument("--jsonl", default=str(PILOT / "contoh_latih.jsonl"))
    ap.add_argument("--output-dir", help="train: folder run BARU (bukan folder model)")
    ap.add_argument("--run-dir", help="reload: folder run hasil train")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--n-frame", type=int, default=5)
    ap.add_argument("--tanpa-potong-pita", action="store_true")
    ap.add_argument("--steps", type=int, default=8, help="jumlah update optimizer yang berhasil")
    ap.add_argument("--maks-percobaan", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-ae", action="store_true", help="juga generasi AE terkontrol (soal acuan) sebelum/sesudah")
    ap.add_argument("--lanjut-tanpa-baseline", action="store_true", help="lanjut training walau baseline gagal")
    ap.add_argument("--sin-cudnn", action="store_true")
    a = ap.parse_args()
    a.potong = not a.tanpa_potong_pita
    if a.mode == "train":
        if not a.output_dir:
            ap.error("--output-dir wajib untuk mode train")
        mode_train(a)
    else:
        if not a.run_dir:
            ap.error("--run-dir wajib untuk mode reload")
        mode_reload(a)


if __name__ == "__main__":
    main()
