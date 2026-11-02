# Pipeline training (Transformers + PEFT)

Kode di bawah adalah **kerangka yang belum dijalankan**. Setiap pemanggilan API harus dicek terhadap versi Transformers/PEFT yang terpasang. Kalau ada yang berbeda, ikuti versi yang terpasang dan catat perbedaannya.

## 1. Memuat model dan processor

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_DIR = ...  # dari konfigurasi lingkungan, lihat lingkungan.md
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
    # opsi: llm_int8_skip_modules=["visual", "lm_head"]  (seperti 2U1; bandingkan VRAM)
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR, quantization_config=bnb, dtype=compute_dtype,
    attn_implementation="sdpa", device_map={"": 0},
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model.config.use_cache = False
```

- Notebook Colab sebelumnya memakai `Qwen3_5ForConditionalGeneration` secara langsung. 2U1 memakai `AutoModelForImageTextToText`. Pastikan `type(model).__name__` sama dengan arsitektur di config.
- Cek: `print(type(model))`, lalu `[n for n, _ in model.named_modules() if "mtp" in n][:5]` untuk melihat apakah modul `mtp` ikut dimuat.

## 2. Membekukan dan memasang LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# Pilihan (B) dari arsitektur-checkpoint.md; regex di-fullmatch oleh PEFT pada nama modul
TARGET = (r"model\.language_model\.layers\.\d+\."
          r"(self_attn\.(q|k|v|o)_proj"
          r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
          r"|mlp\.(gate|up|down)_proj)")

lora = LoraConfig(r=..., lora_alpha=..., lora_dropout=..., bias="none",
                  target_modules=TARGET, task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
```

Pemeriksaan wajib setelah LoRA dipasang:

```python
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
assert trainable and all("lora_" in n for n in trainable), trainable[:10]
assert not any("visual" in n for n in trainable)
assert not any(".mtp." in n or n.startswith("mtp") for n in trainable)
lapisan = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in trainable if ".layers." in n})
print("lapisan terkena LoRA:", lapisan)  # pilihan (B): 0..31; pilihan (A): 3,7,...,31
```

Nilai `r`, `lora_alpha`, `lora_dropout`, dan learning rate ditentukan dalam pilot. Nilai r=4, alpha=8 dari LoRA teks sebelumnya **bukan** konfigurasi final.

## 3. Membangun satu contoh (masking label)

Pola yang dipakai sama dengan 2U1 untuk Qwen3.5: tokenisasi **prompt** (dengan gambar) dan **jawaban** secara terpisah, lalu gabungkan.

```python
from PIL import Image

def bangun_contoh(messages, target_text, image_paths, max_side=None):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    # opsional: perkecil gambar di sini untuk membatasi token visual (catat nilai max_side)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # prompt_text berakhir dengan "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    p = processor(text=[prompt_text], images=images, return_tensors="pt")
    resp_ids = processor.tokenizer(target_text + "<|im_end|>\n",
                                   add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([p.input_ids, resp_ids], dim=1)[0]
    labels = torch.cat([torch.full_like(p.input_ids, -100), resp_ids], dim=1)[0]
    contoh = dict(input_ids=input_ids, labels=labels,
                  attention_mask=torch.ones_like(input_ids),
                  pixel_values=p.pixel_values, image_grid_thw=p.image_grid_thw)
    if "mm_token_type_ids" in p:   # 2U1 meneruskan ini untuk Qwen3.5; cek apakah processor menghasilkannya
        contoh["mm_token_type_ids"] = torch.cat(
            [p.mm_token_type_ids, torch.zeros_like(resp_ids)], dim=1)[0]
    return contoh
```

Pemeriksaan wajib per sampel (jalankan pada beberapa contoh sebelum training):

```python
cfg = model.config  # atau base model sebelum PEFT
n_img_tok = (contoh["input_ids"] == cfg.image_token_id).sum().item()
merge = processor.image_processor.merge_size
assert n_img_tok == (contoh["image_grid_thw"].prod(-1).sum().item() // merge**2)  # gambar benar-benar masuk
assert n_img_tok > 0 and contoh["pixel_values"].numel() > 0
assert (contoh["labels"] != -100).sum() > 0                           # ada token target
# konsistensi template: render penuh harus sama dengan prompt + target
full = processor.apply_chat_template(
    messages + [{"role": "assistant", "content": target_text}], tokenize=False)
assert full == prompt_text + target_text + "<|im_end|>\n", "template latih ≠ inferensi"
print(processor.tokenizer.decode(contoh["input_ids"][contoh["labels"] != -100]))  # harus = target
```

Kalau assertion konsistensi gagal, **berhenti** dan selidiki penyebabnya. Jangan memaksakan training.

## 4. Collator

- Padding ke kanan: `input_ids` diisi `pad_token_id`, `labels` diisi `-100`, `attention_mask` diisi 0, `mm_token_type_ids` (jika ada) diisi 0.
- `pixel_values` dan `image_grid_thw` **digabung (concat) di dimensi 0** antarsampel, bukan di-stack.
- Gunakan `remove_unused_columns=False` bila memakai `Trainer`, supaya `pixel_values` tidak dibuang.
- Mulai dengan batch size 1 + gradient accumulation. Batch > 1 baru dicoba setelah padding terbukti benar.

## 5. Uji 1 batch forward/backward

```python
model.train()
batch = {k: v.to(model.device) for k, v in collator([contoh]).items()}
torch.cuda.reset_peak_memory_stats()
out = model(**batch)
assert torch.isfinite(out.loss), out.loss
out.loss.backward()
grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
print("loss", out.loss.item(), "peak VRAM GiB", torch.cuda.max_memory_allocated() / 2**30)
```

- Ablation cepat untuk membuktikan gambar berpengaruh: hitung loss sampel yang sama dengan frame asli dan dengan frame diganti gambar kosong. Loss yang identik menandakan gambar tidak terpakai. Ini pemeriksaan sanity, bukan evaluasi.

## 6. Training

- `Trainer` + collator buatan sendiri; atau TRL `SFTTrainer` **hanya jika** penanganan gambarnya sudah diverifikasi.
- Gradient checkpointing aktif. Simpan checkpoint secara berkala. Log loss latih dan validasi.
- Untuk T4 (tanpa bf16): `fp16=True`. Untuk GPU yang mendukung bf16: `bf16=True`. Catat yang dipakai.
- Catat waktu per langkah, peak VRAM, dan setiap OOM beserta konfigurasi saat itu.

## 7. Simpan, muat ulang, artefak

Simpan ke folder berstempel waktu di luar `Model/`:
- adapter (`model.save_pretrained`), processor, `config_run.json` (semua hyperparameter, `max_side`/`max_pixels`, jumlah frame, seed, ID dan path checkpoint dasar, opsi target LoRA);
- `pip freeze` dan info GPU; `splits.json`; log training;
- keluaran generasi **mentah**, terpisah dari hasil parsing JSON. Kegagalan parsing dicatat, tidak dibuang.

Uji muat ulang di **sesi/proses baru**: muat model dasar 4-bit + `PeftModel.from_pretrained(adapter)`, generasi greedy dengan `max_new_tokens` yang cukup, lalu bandingkan dengan keluaran sebelum disimpan.

Untuk generasi, gunakan prompt dari fungsi yang sama dengan training (`add_generation_prompt=True, enable_thinking=False`). Kalau keluaran terpotong, naikkan `max_new_tokens` dulu sebelum menyimpulkan model gagal.
