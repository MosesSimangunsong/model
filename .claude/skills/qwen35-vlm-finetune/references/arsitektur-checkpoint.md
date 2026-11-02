# Fakta checkpoint `Model/Qwen3.5-9B`

Dibaca 2 November 2026 dari `config.json`, `preprocessor_config.json`, `chat_template.jinja`, `model.safetensors.index.json`, dan header `.safetensors` (bentuk tensor), **tanpa memuat bobot**.

Nama di bawah adalah **kunci bobot di checkpoint**. Sebelum memasang LoRA, tetap cocokkan dengan `model.named_modules()` pada versi Transformers yang terpasang, karena implementasi dapat memetakan nama secara berbeda.

## Identitas

- `architectures`: `Qwen3_5ForConditionalGeneration`; `model_type`: `qwen3_5`.
- Processor: `Qwen3VLProcessor`; image processor: `Qwen2VLImageProcessorFast`.
- `transformers_version` yang tercatat di config: `4.57.0.dev0`. Colab sebelumnya memakai Transformers 5.22.1.
- Dtype bobot: BF16.

## Bagian bahasa: hibrida, bukan Transformer standar

- 32 lapisan (`num_hidden_layers`), `hidden_size` 4096, `intermediate_size` 12288.
- `layer_types` berpola 3× `linear_attention` lalu 1× `full_attention` (`full_attention_interval: 4`). Akibatnya hanya **8 lapisan full-attention**, yaitu lapisan **3, 7, 11, 15, 19, 23, 27, 31**. Sisanya, 24 lapisan, memakai linear attention.

Modul linear per jenis lapisan:

| Jenis lapisan | Modul (prefiks `model.language_model.layers.N.`) | Bentuk bobot (out × in) |
|---|---|---|
| full_attention | `self_attn.q_proj` | 8192 × 4096 (`attn_output_gate: true`, query + gate) |
| | `self_attn.k_proj`, `self_attn.v_proj` | 1024 × 4096 |
| | `self_attn.o_proj` | ada; bentuk tidak dicek |
| linear_attention | `linear_attn.in_proj_qkv` | 8192 × 4096 |
| | `linear_attn.out_proj` | 4096 × 4096 |
| | `linear_attn.in_proj_z`, `in_proj_a`, `in_proj_b` | ada; bentuk tidak dicek |
| | `linear_attn.conv1d` | konvolusi, **bukan** Linear |
| semua lapisan | `mlp.gate_proj` (12288 × 4096), `mlp.up_proj`, `mlp.down_proj` | |

### Konsekuensi untuk konfigurasi LoRA Colab sebelumnya

Konfigurasi notebook menargetkan modul yang mengandung `.self_attn.` dan berakhiran `q_proj`/`v_proj`, dengan r=4. Hitungannya:

- q_proj: 4 × (4096 + 8192) = 49.152 parameter
- v_proj: 4 × (4096 + 1024) = 20.480 parameter
- Total per lapisan: 69.632. Dikali **8 lapisan = 557.056**.

Angka ini **persis sama** dengan jumlah parameter dapat dilatih yang dilaporkan notebook. Jadi LoRA teks itu hanya mengenai 8 lapisan full-attention. Linear attention dan MLP tidak tersentuh, dan modul `mtp` juga tidak.

### Pilihan target LoRA (keputusan pilot, catat yang dipilih)

- **(A) Seperti sebelumnya**: `self_attn.q_proj`/`v_proj`, hanya 8 lapisan. Murah, tetapi cakupannya sempit.
- **(B) Semua proyeksi linear bagian bahasa**: `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (dengan atau tanpa `in_proj_a`/`in_proj_b`), dan `mlp.{gate,up,down}_proj` di 32 lapisan. Kecualikan `visual`, `lm_head`, `embed_tokens`, dan `mtp`.

Usulan: (B) untuk pilot multimodal, dengan alasan cakupannya menyeluruh. Ini **usulan**, bukan hasil. Bandingkan jumlah parameter dan VRAM-nya sebelum memutuskan.

Jangan memakai `target_modules="all-linear"` tanpa pengecualian, karena akan ikut mengenai modul `visual`.

## Modul lain

- `mtp.*` (multi-token prediction): bobotnya ada di checkpoint dengan `self_attn.q_proj` dan sebagainya. Apakah kelas Transformers memuat modul ini **belum diketahui**. Periksa dengan `named_modules()` dan kecualikan dari LoRA.
- `lm_head` tidak di-tie dengan embedding (`tie_word_embeddings: false`).

## Vision encoder (`model.visual`)

- 27 blok, `hidden_size` 1152, `patch_size` 16, `spatial_merge_size` 2, `temporal_patch_size` 2, `out_hidden_size` 4096.
- Modul: `blocks.N.attn.qkv` (3456 × 1152), `blocks.N.attn.proj`, `blocks.N.mlp.linear_fc1`/`linear_fc2`, `merger.linear_fc1`/`linear_fc2`, `patch_embed.proj`, `pos_embed`.
- Penamaannya berbeda dari bagian bahasa, sehingga target berbasis nama seperti `q_proj` atau `down_proj` tidak mengenai vision.
- Sebagai pembanding, 2U1 tidak mengkuantisasi `visual` dan `lm_head` (`llm_int8_skip_modules=["visual","lm_head"]`). Ini opsi yang perlu dipertimbangkan, bukan keputusan.

## Token khusus (dari config.json)

| Token | ID |
|---|---|
| `<\|image_pad\|>` (`image_token_id`) | 248056 |
| `<\|video_pad\|>` (`video_token_id`) | 248057 |
| `<\|vision_start\|>` | 248053 |
| `<\|vision_end\|>` | 248054 |
| EOS bagian teks (`text_config.eos_token_id`) | 248044 |

Di kode, ambil nilai ini dari `model.config` atau processor. Jangan di-hardcode.

## Jumlah token gambar

- Satu token visual mewakili area 32 × 32 piksel (patch 16 × merge 2).
- Batas default processor: `shortest_edge` 65.536 piksel (≈256²) dan `longest_edge` 16.777.216 piksel (≈4096²). Frame 1080p **tidak diperkecil** secara default.
- Frame pilot berukuran **1804 × 1080**. Perkiraan dengan pembulatan ke kelipatan 32 adalah 56 × 34 ≈ **1.900 token per frame**, sehingga 27 frame ≈ **51.000 token visual**. Ini **perkiraan**. Ukur dengan `image_grid_thw` (token = `t·h·w / 4`).
- Implikasinya: jumlah frame dan resolusi harus dibatasi (atur `max_pixels`/`size`, atau kecilkan gambar sebelum masuk processor), lalu cek apakah tulisan formula masih terbaca. Nilainya ditentukan lewat pilot.

## Chat template

- Gambar dirender sebagai `<|vision_start|><|image_pad|><|vision_end|>`. Processor lalu menggandakan `<|image_pad|>` sesuai `image_grid_thw`.
- Opsi `add_vision_id` menambahkan teks `Picture N: ` sebelum tiap gambar.
- Prompt generasi dengan `enable_thinking=False` berakhir dengan `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- Saat percakapan berisi jawaban assistant (data latih) tanpa `reasoning_content`, template merender **scaffold yang sama** (`<think>\n\n</think>\n\n`) sebelum isi jawaban. Jadi bentuk latih dan inferensi konsisten.
- **Terverifikasi** dengan merender `chat_template.jinja` asli (jinja2, 2 November 2026): render percakapan penuh = render prompt (`add_generation_prompt=True, enable_thinking=False`) + target + `<|im_end|>\n`.
- Scaffold itu termasuk **prompt**: labelnya `-100`. Loss dimulai dari isi jawaban, sama seperti cara 2U1 menangani Qwen3.5 tanpa reasoning.
- Template melakukan `trim` pada isi pesan. Spasi atau baris baru di awal dan akhir target akan hilang, jadi simpan target tanpa spasi tepi.
- Pesan system tidak boleh berisi gambar.

## Peringatan dari Colab sebelumnya

Kernel opsional `causal_conv1d` dan `flash-linear-attention` tidak tersedia, sehingga linear attention memakai implementasi PyTorch (lebih lambat). Periksa kompatibilitasnya sebelum memasang kernel tersebut. 2U1 juga melaporkan Flash Attention 2 bermasalah untuk Qwen3.5; pakai `sdpa`.
