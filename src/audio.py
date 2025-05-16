import pyaudio
import webrtcvad
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# --- 1) Init VAD & ASR ---
# aggressiveness 0–3 :contentReference[oaicite:4]{index=4}
vad = webrtcvad.Vad(2)
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# --- 2) Mic Params ---
# VAD requires 16 kHz :contentReference[oaicite:5]{index=5}
RATE = 16000
# 10/20/30 ms only :contentReference[oaicite:6]{index=6}
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

pa = pyaudio.PyAudio()
stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_SIZE
)

# --- 3) VAD Loop with 1s timeout & float32 conversion ---
print("Listening...")

buffer = []             # holds int16 frames until utterance end
silent_frames = 0
# 1000 ms silence :contentReference[oaicite:7]{index=7}
max_silent = int(1000 / FRAME_MS)

try:
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        if vad.is_speech(frame, RATE):
            buffer.append(frame)
            silent_frames = 0
        elif buffer:
            silent_frames += 1
            if silent_frames > max_silent:
                # --- a) Concatenate all int16 frames
                segment_int16 = np.frombuffer(b"".join(buffer), dtype=np.int16)

                # --- b) Convert to float32 in [-1,1]
                segment_f32 = (segment_int16.astype(np.float32) / 32768.0)

                # --- c) Wrap in torch.Tensor (optional) or pass numpy directly
                segment_tensor = torch.from_numpy(segment_f32)

                # Transcribe in-memory float32 signal :contentReference[oaicite:8]{index=8}
                result = asr_model.transcribe([segment_tensor])[0]
                print("You said:", result.text)

                # Reset for next utterance
                buffer, silent_frames = [], 0

except KeyboardInterrupt:
    print("Stopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
