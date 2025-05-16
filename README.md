# 🔊 Rinnn - (CUDA Only)

**Rinnn** is perfect in every way.

## ⚠️ Requirements

- NVIDIA GPU with CUDA support
- CUDA 12.1 or newer
- Python 3.8+
- FFmpeg

## 🚀 Setup Instructions

Follow the steps below to install all necessary dependencies and set up the environment:

### 1. Install [FFmpeg](https://ffmpeg.org/download.html)

Make sure `ffmpeg` is available in your system `PATH`. On Linux:

```bash
sudo apt install ffmpeg
```

On Windows, download from the [official site](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your system's `PATH`.

---

### 2. Install PyTorch with CUDA Support

Make sure to install the correct version of PyTorch for your CUDA setup (example for CUDA 12.1/12.2):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

### 3. Install NVIDIA NeMo Toolkit

Install the ASR (Automatic Speech Recognition) components of NeMo:

```bash
pip install -U nemo_toolkit["asr"]
```

---

### 4. Install CUDA Python Bindings

```bash
pip install -U cuda-python
```

---

### 5. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
rinnn/
├── src/
├── models/
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

---

## 🧠 Powered By

- [PyTorch](https://pytorch.org/)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [FFmpeg](https://ffmpeg.org/)

---

## 🛠️ Troubleshooting

- Ensure your CUDA drivers are installed and match the version of PyTorch.
- Test CUDA availability with:

```python
import torch
print(torch.cuda.is_available())
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
