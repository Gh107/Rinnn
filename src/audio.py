import pyaudio
import webrtcvad
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Constants
RATE = 16000              # Sampling rate for VAD
FRAME_MS = 30             # Duration of each frame in ms (10/20/30 ms allowed)
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Initialize VAD and ASR model
vad = webrtcvad.Vad(2)  # aggressiveness: 0-3
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)


def listen_and_transcribe(max_silence_ms: int = 1000):
    """
    Listen to microphone input, perform VAD to detect speech segments,
    and transcribe each utterance with the ASR model.

    Parameters:
    - max_silence_ms: maximum silence (in ms) to wait before ending an utterance.

    Yields:
    - result.text: the transcribed text of each detected utterance.
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE
    )

    buffer = []  # holds int16 frames until utterance end
    silent_frames = 0
    max_silent = int(max_silence_ms / FRAME_MS)

    print("VAD initialized.")
    result = None
    try:
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            if vad.is_speech(frame, RATE):
                buffer.append(frame)
                silent_frames = 0
            elif buffer:
                silent_frames += 1
                if silent_frames > max_silent:
                    # Concatenate and convert
                    segment_int16 = np.frombuffer(
                        b"".join(buffer), dtype=np.int16)
                    segment_f32 = segment_int16.astype(np.float32) / 32768.0
                    segment_tensor = torch.from_numpy(segment_f32)

                    # Transcribe
                    result = asr_model.transcribe([segment_tensor])[0]
                    yield result.text

                    # Reset for next utterance
                    buffer.clear()
                    silent_frames = 0
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
