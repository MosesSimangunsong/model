"""Uji unit latih_pilot.py yang bisa jalan di laptop (CPU, tanpa model 9B).

Pakai: python scripts/uji_unit_latih_pilot.py
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latih_pilot as lp  # noqa: E402
import uji_1_batch as u  # noqa: E402

hasil = []


def uji(nama, kondisi):
    hasil.append((nama, bool(kondisi)))
    print(("LOLOS " if kondisi else "GAGAL ") + nama)


# 1. parsing ketat: JSON valid, rusak, berpagar Markdown (tidak diperbaiki)
uji("parse JSON valid", lp.parse_ketat('{"a": 1}')["parse_status"] == "terbaca")
p = lp.parse_ketat('{"a": 1,')
uji("parse JSON rusak -> tidak_terbaca + galat", p["parse_status"] == "tidak_terbaca" and p["galat"])
p = lp.parse_ketat('```json\n{"a": 1}\n```')
uji("pagar Markdown dicatat dan TIDAK diperbaiki", p["berpagar_markdown"] and p["parse_status"] == "tidak_terbaca")

# 2. pemetaan AE ke soal QG
rows = [json.loads(l) for l in open(lp.PILOT / "contoh_latih.jsonl", encoding="utf-8")]
uji("pemetaan AE data pilot benar", lp.periksa_pemetaan_ae(rows) == [])
salah = copy.deepcopy(rows)
salah[1]["input_extra"]["answer_key"] = "B"
uji("AE dengan kunci berbeda dari QG ditolak", any("berbeda" in g for g in lp.periksa_pemetaan_ae(salah)))
salah = copy.deepcopy(rows)
t = json.loads(salah[2]["target"]); t["question_id"] = "q9"; salah[2]["target"] = json.dumps(t)
uji("AE dengan question_id salah ditolak", any("question_id" in g for g in lp.periksa_pemetaan_ae(salah)))

# 3. evidence yang tidak ditampilkan ditolak
manifest = json.loads((lp.PILOT / "segmen.json").read_text(encoding="utf-8"))
try:
    u.bangun(None, rows[0], dict(manifest, frames=manifest["frames"][:-1]), 5, 1024, True)
    uji("evidence F027 tidak ditampilkan ditolak", False)
except AssertionError as e:
    uji("evidence F027 tidak ditampilkan ditolak", "F027" in str(e))

# 4. deteksi update dilewati
uji("norm inf -> dilewati", lp.status_langkah(float("inf"), 1024.0, 1024.0, True) == "dilewati")
uji("scale turun -> dilewati", lp.status_langkah(1.0, 1024.0, 512.0, True) == "dilewati")
uji("norm finite, scale tetap -> update", lp.status_langkah(1.0, 1024.0, 1024.0, True) == "update")
uji("tanpa scaler (bf16), norm finite -> update", lp.status_langkah(1.0, 1.0, 1.0, False) == "update")

# 4b. jalur nyata GradScaler: gradien inf membuat step dilewati dan parameter tidak berubah
try:
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([w], lr=0.1)
    sc = torch.amp.GradScaler("cpu", init_scale=1024.0)
    (sc.scale((w * float("inf")).sum())).backward()
    sc.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_([w], 1.0)
    lama = sc.get_scale(); sc.step(opt); sc.update(); baru = sc.get_scale()
    uji("GradScaler: gradien inf -> step dilewati, bobot tetap, terdeteksi",
        torch.equal(w.detach(), torch.ones(3)) and lp.status_langkah(norm, lama, baru, True) == "dilewati")
except Exception as e:  # GradScaler CPU tidak tersedia di semua versi torch
    print("LEWATI uji GradScaler CPU:", type(e).__name__, e)

# 5. loss hemat (logits_to_keep) == loss standar (shift label), pada model palsu kecil
class ModelPalsu(torch.nn.Module):
    def __init__(self, v=11, d=6):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.head = torch.nn.Embedding(v, d), torch.nn.Linear(d, v)

    def forward(self, input_ids, labels=None, logits_to_keep=0, **_):
        h = torch.cumsum(self.emb(input_ids), dim=1)  # kausal: posisi t hanya melihat <= t
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        logits = self.head(h)
        loss = None
        if labels is not None:  # setara ForCausalLMLoss: geser satu posisi, abaikan -100
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                                                     labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


ids = torch.tensor([[1, 4, 2, 7, 3, 9, 5, 8]])
n = 3  # 3 token terakhir = target
lab = torch.full_like(ids, -100); lab[:, -n:] = ids[:, -n:]
batch = {"input_ids": ids, "labels": lab}
m = ModelPalsu()
hemat, standar = u.hitung_loss(m, batch, n, True).item(), u.hitung_loss(m, batch, n, False).item()
uji(f"loss hemat == standar pada model palsu ({hemat:.6f} vs {standar:.6f})", abs(hemat - standar) < 1e-6)

# 6. rata-rata berbobot token
pc = [{"task": "qg", "loss": 1.0, "token_target": 100}, {"task": "ae", "loss": 2.0, "token_target": 300}]
uji("loss berbobot token = 1,75", abs(lp.loss_berbobot(pc) - 1.75) < 1e-12)

gagal = [n for n, ok in hasil if not ok]
print(f"\n{len(hasil) - len(gagal)}/{len(hasil)} lolos")
sys.exit(1 if gagal else 0)
