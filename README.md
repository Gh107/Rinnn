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
pip install deepspeed

### 5. Install Remaining Dependencies

```bash
pip install nemo_toolkit["asr"] cuda-python langchain langgraph langchain-community langchain-mcp-adapters langchain-openai langchain-huggingface langchain-chroma beautifulsoup4 webrtcvad pyaudio soundfile huggingface_hub[cli] sounddevice
```
---

### 6. Install [Index-TTS](https://github.com/index-tts/index-tts?tab=readme-ov-file)

```bash
git clone https://github.com/index-tts/index-tts.git
pip install -r requirements.txt
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
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
