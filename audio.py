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
# Audio setup with fallback flag
try:
    pa = pyaudio.PyAudio()
    pa.get_default_input_device_info()
    mic_available = True
except Exception:
    print("[Warning] Microphone not available. Falling back to text input.")
    mic_available = False


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

    print("She's listening...")
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
<<<<<<< HEAD
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
        exit()
    except Exception as e:
        print(f"An error occurred during listening/transcription: {e}")
=======
>>>>>>> parent of 73f463a (Removed pyaudio)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
