import sounddevice as sd
import webrtcvad
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Constants
RATE = 16000              # Sampling rate (Hz)
# Duration of each frame in ms (10, 20, or 30 ms allowed)
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # Samples per frame
# Note: FORMAT and CHANNELS are implicitly handled by sounddevice parameters
CHANNELS = 1

# Initialize VAD and ASR model
vad = webrtcvad.Vad(2)  # aggressiveness: 0-3
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

# Audio setup with fallback flag using sounddevice
mic_available = False
try:
    # Check if a default input device can be opened with the desired settings
    sd.check_input_settings(samplerate=RATE, channels=CHANNELS, dtype='int16')
    mic_available = True
except Exception as e:
    print(
        f"[Warning] Microphone not available or configuration issue: {e}. Falling back to text input.")
    mic_available = False


def listen_and_transcribe(max_silence_ms: int = 1500):
    """
    Listen to microphone input using sounddevice, perform VAD to detect speech segments,
    and transcribe each utterance with the ASR model.

    Parameters:
    - max_silence_ms: maximum silence (in ms) to wait before ending an utterance.

    Yields:
    - result_text: the transcribed text of each detected utterance.
    """
    if not mic_available:
        print("Microphone not available. Cannot start listening.")
        return  # Exit the generator if no mic

    # Open the sounddevice InputStream
    stream = sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype='int16',        # Equivalent to pyaudio.paInt16
        blocksize=FRAME_SIZE  # Equivalent to frames_per_buffer
    )
    stream.start()  # Explicitly start the stream

    buffer = []  # holds bytes frames until utterance end
    silent_frames = 0
    max_silent = int(max_silence_ms / FRAME_MS)

    print("She's listening...")
    # result = None # Not strictly necessary before the loop
    try:
        while True:
            # Read FRAME_SIZE samples; returns (numpy_array, overflow_flag)
            frame_data, overflow = stream.read(FRAME_SIZE)
            if overflow:
                print("[Warning] Input overflow detected.")

            # Convert numpy array (int16) to bytes for VAD
            frame = frame_data.tobytes()

            if vad.is_speech(frame, RATE):
                buffer.append(frame)
                silent_frames = 0
            # If not speech, but buffer has data (i.e., we were speaking)
            elif buffer:
                silent_frames += 1
                if silent_frames > max_silent:
                    # End of utterance: Concatenate and convert
                    segment_bytes = b"".join(buffer)
                    segment_int16 = np.frombuffer(
                        segment_bytes, dtype=np.int16)
                    segment_f32 = segment_int16.astype(np.float32) / 32768.0
                    segment_tensor = torch.from_numpy(segment_f32)

                    # Transcribe
                    # ASRModel.transcribe returns a list (batch size 1),
                    # so we take the first element.
                    # It usually returns a Hypothesis object with a .text attribute.
                    transcription_results = asr_model.transcribe(
                        [segment_tensor])

                    # Extract text, handling potential differences in NeMo versions/configs
                    result_text = ""
                    if transcription_results:
                        first_result = transcription_results[0]
                        if hasattr(first_result, 'text'):
                            result_text = first_result.text
                        elif isinstance(first_result, str):
                            result_text = first_result

                    if result_text:  # Only yield if transcription is not empty
                        yield result_text

                    # Reset for next utterance
                    buffer.clear()
                    silent_frames = 0
    finally:
        print("Stopping stream...")
        stream.stop()
        stream.close()
