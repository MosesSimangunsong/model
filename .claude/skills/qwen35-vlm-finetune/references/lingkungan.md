# Lingkungan: laptop, Colab, GPU IT Del

## Yang sudah diketahui

| Lingkungan | Status | Peran |
|---|---|---|
| Laptop: Windows 11, RTX 3060 Laptop **6 GB**, venv `.venv`, FFmpeg, faster-whisper (CPU) | Terverifikasi | Prapemrosesan, anotasi, menulis/debug kode, cek tokenisasi tanpa memuat bobot |
| Google Colab | Bisa diakses; **jenis GPU & langganan belum dikonfirmasi** | Inferensi dan training pilot 9B |
| GPU IT Del (B200) | Belum dapat diakses | Eksperimen utama nanti |

Fakta referensi: di Colab T4 (14,6 GiB), model 4-bit setelah dimuat memakai ~7,35 GiB. Itu memori **setelah memuat**, bukan puncak training. Gambar menambah aktivasi secara signifikan.

## Cek wajib di awal setiap sesi (catat hasilnya)

```python
import torch, platform, transformers, peft, bitsandbytes, accelerate
print(platform.platform(), torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB")
print("bf16:", torch.cuda.is_bf16_supported())
print(transformers.__version__, peft.__version__, bitsandbytes.__version__, accelerate.__version__)
```

Jangan menentukan batch size, jumlah frame, resolusi, atau rank sebelum hasil cek ini ada.

## Portabilitas path

- Satu sumber konfigurasi path, misalnya variabel lingkungan atau satu file `config` kecil. Pakai `pathlib.Path`.
  - `LEVELEARN_ROOT`: root data. Laptop: `E:\Semester 7\TA\Latihan_Model`; Colab: folder di `/content/drive/MyDrive/...`.
  - `QWEN_MODEL_DIR`: folder checkpoint. Laptop: `Model/Qwen3.5-9B`; Colab: salinan di Drive (path sebelumnya `/content/drive/MyDrive/LeveLearn/models/Qwen3.5-9B`).
- Sel khusus Colab (`drive.mount`) hanya dijalankan bila terdeteksi di Colab (`"google.colab" in sys.modules`). Jangan mount ulang bila path sudah dapat diakses.
- `%pip` di notebook dan `python -m pip` di terminal itu berbeda. Jangan dicampur.
- Output (adapter, log) selalu ke lokasi permanen (Drive/disk), bukan `/content` yang hilang saat sesi habis.

## Yang bisa dikerjakan di laptop tanpa GPU besar

- Membangun dataset, memeriksa template, dan memeriksa masking: cukup `AutoProcessor`, tanpa memuat bobot.
- Menghitung jumlah token visual per konfigurasi resolusi/frame dengan `image_grid_thw`.
- Memvalidasi JSON keluaran dan manifest split.
