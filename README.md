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

On Windows, download from the [official site](https://ffmpeg.org/download.html), extract it to the root of the project.

---

### 2. Install PyTorch with CUDA Support

Make sure to install the correct version of PyTorch for your CUDA setup (example for CUDA 12.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---
sudo apt-get install libportaudio2
---
pip install kokoro
---

### 5. Install Remaining Dependencies

```bash
pip install nemo_toolkit["asr"] cuda-python langchain langgraph langchain-community langchain-openai langchain-huggingface langchain-chroma beautifulsoup4 huggingface_hub[cli] sounddevice silero-vad
```
---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
