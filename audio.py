import sounddevice as sd
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Constants
RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
CHANNELS = 1

# --- Initialize Silero VAD and ASR model ---

# Load Silero VAD model
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False)
    # The 'utils' tuple contains several functions, but we only need the model itself
    # for frame-by-frame processing and get_speech_timestamps if needed for other uses.
    # No need to unpack get_speech_timestamps, etc. if not directly used here.
    # Keeping this line to ensure utils is unpacked if other parts of code implicitly use them
    (get_speech_timestamps, save_audio, read_audio,
     VADIterator, collect_chunks) = utils

    vad_model.eval()  # Set the VAD model to evaluation mode
    print("Silero VAD model loaded successfully.")

except Exception as e:
    print(f"[ERROR] Could not load Silero VAD model: {e}")
    exit()

# Initialize ASR model
try:
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    print("NeMo ASR model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load NeMo ASR model: {e}")
    exit()

# Audio setup with fallback flag using sounddevice
mic_available = False
try:
    sd.check_input_settings(samplerate=RATE, channels=CHANNELS, dtype='int16')
    mic_available = True
    print("Microphone detected and configured.")
except Exception as e:
    print(
        f"[Warning] Microphone not available or configuration issue: {e}. Falling back to text input.")
    mic_available = False


def listen_and_transcribe(max_silence_ms: int = 1500, vad_threshold: float = 0.5):
    """
    Listen to microphone input using sounddevice, perform Silero VAD to detect speech segments,
    and transcribe each utterance with the ASR model.

    Parameters:
    - max_silence_ms: maximum silence (in ms) to wait before ending an utterance.
    - vad_threshold: The probability threshold for Silero VAD to consider a frame as speech (0.0 to 1.0).

    Yields:
    - result_text: the transcribed text of each detected utterance.
    """
    if not mic_available:
        print("Microphone not available. Cannot start listening.")
        return

    stream = sd.InputStream(
        samplerate=RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=FRAME_SIZE
    )
    stream.start()

    audio_buffer = []
    silent_frames = 0
    max_silent = int(max_silence_ms / FRAME_MS)

    # NO vad_state variable needed here, as the model manages its own state
    # vad_state = None # REMOVE THIS LINE

    print("She's listening...")
    try:
        while True:
            frame_data, overflow = stream.read(FRAME_SIZE)
            if overflow:
                print("[Warning] Input overflow detected.")

            # --- CRITICAL CHANGE HERE ---
            # Ensure frame_data is flattened to 1D before converting to tensor
            # This handles cases where sounddevice might return (FRAME_SIZE, 1) for mono audio.
            frame_f32 = (frame_data.flatten()).astype(np.float32) / 32768.0

            # Now, frame_f32 is guaranteed to be 1D, so unsqueeze(0) will result in a 2D tensor [1, samples]
            frame_tensor = torch.from_numpy(frame_f32).unsqueeze(0)

            speech_prob_tensor = vad_model(frame_tensor, RATE)
            speech_prob = speech_prob_tensor.item()

            if speech_prob > vad_threshold:
                audio_buffer.append(frame_data)
                silent_frames = 0
            elif audio_buffer:
                silent_frames += 1
                if silent_frames > max_silent:
                    segment_int16 = np.concatenate(audio_buffer).flatten()
                    segment_f32 = segment_int16.astype(np.float32) / 32768.0
                    segment_tensor = torch.from_numpy(segment_f32)

                    transcription_results = asr_model.transcribe(
                        [segment_tensor])

                    result_text = ""
                    if transcription_results:
                        first_result = transcription_results[0]
                        if hasattr(first_result, 'text'):
                            result_text = first_result.text
                        elif isinstance(first_result, str):
                            result_text = first_result

                    if result_text:
                        yield result_text

                    audio_buffer.clear()
                    silent_frames = 0
                    vad_model.reset_states()

    finally:
        print("Stopping stream...")
        stream.stop()
        stream.close()
