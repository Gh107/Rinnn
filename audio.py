import sounddevice as sd
import webrtcvad
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

# Constants
RATE = 16000              # Sampling rate (Hz)
# Duration of each frame in ms (10, 20, or 30 ms for VAD)
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # Samples per frame
# For sounddevice, dtype='int16' will be used. CHANNELS=1.
CHANNELS = 1

# Initialize VAD and ASR model
try:
    # Aggressiveness: 0 (least aggressive) to 3 (most aggressive)
    vad = webrtcvad.Vad(2)
    # Ensure NeMo and its dependencies are correctly installed.
    # This will download the model if not already cached.
    print("Loading ASR model (nvidia/parakeet-tdt-0.6b-v2)... This might take a moment.")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    print("ASR model loaded.")
except Exception as e:
    print(f"Error initializing VAD or ASR model: {e}")
    print("Please ensure webrtcvad and NeMo ASR toolkit are installed correctly.")
    print("For NeMo, try: pip install nemo_toolkit['asr']")
    # Exit or handle if essential components fail to load
    asr_model = None
    vad = None


# Audio setup with fallback flag using sounddevice
mic_available = False
if asr_model and vad:  # Only check mic if models loaded
    try:
        # Check if a default input device can be opened with the desired settings
        sd.check_input_settings(
            samplerate=RATE, channels=CHANNELS, dtype='int16')
        mic_available = True
        print("Microphone check successful using sounddevice.")
    except Exception as e:
        print(
            f"[Warning] Microphone not available or configuration issue with sounddevice: {e}")
        print("Falling back to indicating no microphone.")
        # mic_available remains False


def listen_and_transcribe(max_silence_ms: int = 1000):
    """
    Listen to microphone input using sounddevice, perform VAD to detect speech segments,
    and transcribe each utterance with the ASR model.

    Parameters:
    - max_silence_ms: maximum silence (in ms) to wait before ending an utterance.

    Yields:
    - transcribed_text: the transcribed text of each detected utterance.
    """
    if not mic_available:
        print("Microphone not available. Cannot start listening.")
        return
    if not asr_model or not vad:
        print("ASR model or VAD not initialized. Cannot start listening.")
        return

    # buffer holds byte frames of int16 audio data until utterance end
    buffer = []
    silent_frames = 0
    max_silent_frames = int(max_silence_ms / FRAME_MS)

    print(f"She's listening (using sounddevice)... (Speak into the default microphone)")
    print(
        f"Waiting for speech. Max silence between speech segments: {max_silence_ms}ms.")

    try:
        # Using sounddevice.InputStream as a context manager
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='int16',        # Data type for the audio samples
            blocksize=FRAME_SIZE  # Number of frames per block/callback
        ) as stream:
            while True:
                # Read FRAME_SIZE samples from the stream.
                # frame_data is a NumPy array of shape (FRAME_SIZE, CHANNELS) and dtype 'int16'.
                frame_data, overflow = stream.read(FRAME_SIZE)

                if overflow:
                    print("[Warning] Input overflow detected during recording.")
                    # Optionally, could clear buffer or take other action if overflow is problematic

                # Convert the NumPy array to bytes for VAD processing.
                # VAD expects raw audio bytes. For int16, each sample is 2 bytes.
                frame_bytes = frame_data.tobytes()

                # Check if the current frame contains speech
                try:
                    is_speech = vad.is_speech(frame_bytes, RATE)
                except Exception as e:
                    # webrtcvad can sometimes raise errors if frame length is incorrect,
                    # though FRAME_SIZE should be valid.
                    print(f"Error in VAD processing: {e}")
                    is_speech = False  # Assume not speech if VAD fails

                if is_speech:
                    buffer.append(frame_bytes)
                    silent_frames = 0
                # If not speech, but buffer has data (i.e., speech just ended)
                elif buffer:
                    silent_frames += 1
                    if silent_frames > max_silent_frames:
                        # End of utterance detected
                        print(
                            f"End of utterance detected after {silent_frames * FRAME_MS}ms of silence.")

                        # Concatenate all byte frames in the buffer
                        full_audio_bytes = b"".join(buffer)

                        # Convert the full audio segment from bytes to a NumPy array of int16
                        segment_int16 = np.frombuffer(
                            full_audio_bytes, dtype=np.int16)

                        # Convert int16 NumPy array to float32 NumPy array, normalized to [-1.0, 1.0]
                        # This is the format expected by many ASR models.
                        segment_f32 = segment_int16.astype(
                            np.float32) / 32768.0

                        # Convert float32 NumPy array to a PyTorch tensor
                        segment_tensor = torch.from_numpy(segment_f32)

                        print(
                            f"Transcribing {len(segment_f32)/RATE:.2f}s audio segment...")
                        # Transcribe the audio tensor
                        # asr_model.transcribe() expects a list of tensors.
                        # It returns a list of Hypothesis objects (by default) or list of strings.
                        transcription_results = asr_model.transcribe(
                            [segment_tensor])

                        # Assuming transcribe returns [HypothesisObject] or similar with a .text attribute
                        # or just [str]
                        if transcription_results:
                            if hasattr(transcription_results[0], 'text'):
                                transcribed_text = transcription_results[0].text
                            else:  # If it's a list of strings
                                transcribed_text = transcription_results[0]

                            if transcribed_text:  # Only yield if not empty
                                print(
                                    f"Transcription result: '{transcribed_text}'")
                                yield transcribed_text
                            else:
                                print("Transcription was empty.")
                        else:
                            print("ASR model returned no transcription.")

                        # Reset buffer and silence counter for the next utterance
                        buffer.clear()
                        silent_frames = 0
                        print("Listening for next utterance...")
                # If not speech and buffer is empty, just continue listening silently

    except KeyboardInterrupt:
        print("\nTranscription stopped by user.")
    except Exception as e:
        print(f"An error occurred during listening/transcription: {e}")
    finally:
        # The 'with stream:' context manager handles stopping and closing the stream.
        print("Audio stream closed.")
