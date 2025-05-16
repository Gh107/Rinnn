import wave
import io
import queue
import threading
import sys

import numpy as np
import pyaudio
import torch
import soundfile as sf

import whisper
from silero_vad import load_silero_vad, get_speech_timestamps

# 1) Load models
vad_model = load_silero_vad()         # Silero VAD
stt_model = whisper.load_model("base")  # Whisper ASR

# 2) Audio settings
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_DURATION_MS = 30
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)

# 3) PyAudio setup
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=FRAME_SIZE)

audio_q = queue.Queue()


def audio_reader():
    """Continuously read raw PCM frames into a queue."""
    print("Audio reader thread started.")
    while True:
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio_q.put(data)


def pcm_bytes_to_float_tensor(pcm_bytes: bytes) -> torch.Tensor:
    """Convert raw int16 PCM bytes to a float32 torch tensor in [-1,1]."""
    # Interpret as little-endian int16
    int_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    float_data = int_data.astype(np.float32) / 32768.0
    return torch.from_numpy(float_data)


def transcribe_loop():
    """Accumulate frames, detect speech, then slice & transcribe in memory."""
    print("Transcription thread started.")
    buffer_bytes = b""
    while True:
        frame = audio_q.get()
        buffer_bytes += frame

        # Convert entire buffer to float tensor for VAD
        wav_tensor = pcm_bytes_to_float_tensor(buffer_bytes)
        # Get speech timestamps in sample indices
        speech_ts = get_speech_timestamps(
            wav_tensor, vad_model, sampling_rate=RATE, return_seconds=False)

        if speech_ts:
            # For each detected speech segment, pull out that slice
            for seg in speech_ts:
                start_sample, end_sample = seg["start"], seg["end"]
                segment_tensor = wav_tensor[start_sample:end_sample]

                # Write segment to an in-memory WAV
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, segment_tensor.numpy(),
                             RATE, format="WAV")
                    wav_buffer.seek(0)

                    # Whisper can accept a file-like object
                    result = stt_model.transcribe(wav_buffer)
                    text = result["text"].strip()
                    if text:
                        print(f"You: {text}")
                        # Here you’d feed `text` into your LangChain agent, e.g.:
                        # reply = agent.run({"user_input": text, ...})
                        # print("Waifu:", reply)

            # Drop processed bytes to avoid reprocessing
            # Find maximum end_sample in bytes:
            max_end = max(seg["end"] for seg in speech_ts)
            byte_cutoff = max_end * 2  # int16 -> 2 bytes/sample
            buffer_bytes = buffer_bytes[byte_cutoff:]


def dump_seconds(n_seconds=5):
    frames = []
    for _ in range(int(RATE/FRAME_SIZE * n_seconds)):
        frames.append(audio_q.get())
    with wave.open("debug.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    print("Wrote debug.wav; play it back to confirm there's audio")


# 4) Launch threads
threading.Thread(target=audio_reader, daemon=True).start()
threading.Thread(target=transcribe_loop, daemon=True).start()
dump_seconds(5)  # Optional: dump a few seconds of audio for debugging
print("🎙️ Listening (memory-only pipeline). Press Ctrl+C to quit.")
try:
    while True:
        pass
except KeyboardInterrupt:
    print("\nStopping…")
    stream.stop_stream()
    stream.close()
    pa.terminate()
    sys.exit()
