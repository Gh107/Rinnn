from datetime import datetime
from timeit import default_timer as timer
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import nemo.collections.asr as nemo_asr

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from personality import personality

from chatterbox.tts import ChatterboxTTS

load_dotenv()

RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
CHANNELS = 1

AUDIO_PROMPT_PATH = "vc.mp3"
EXAGGERATION_FACTOR = 1.0
TEMPERATURE_VALUE = 0.7
CFG_WEIGHT_VALUE = 3.0


class DuckDuckGoSearcher:
    """
    A simple class that encapsulates DuckDuckGo search logic using requests.Session.
    This approach helps to keep code modular, testable, and easily extendable with
    future improvements like caching or concurrency.
    """

    def __init__(self, max_results: int = 5):
        self.url = "https://html.duckduckgo.com/html/"
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search(self, query: str) -> list[dict]:
        """
        Perform a search on DuckDuckGo and parse the top results.

        :param query: The search string.
        :return: A list of dictionaries, each containing 'title', 'url', and 'snippet'.
        """
        try:
            data = {"q": query}
            resp = self.session.post(self.url, data=data, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"DuckDuckGo search request failed: {e}")

        soup = BeautifulSoup(resp.text, "html.parser")
        result_blocks = soup.select(".result__a")[: self.max_results]

        results = []
        for a in result_blocks:
            title = a.get_text(strip=True)
            href = a.get("href", "")
            snippet_tag = a.find_next_sibling("a")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            results.append({
                "title": title,
                "url": href,
                "snippet": snippet
            })
        return results


@tool
def duckduckgo_search(query: str) -> str:
    """
    Searches the web via DuckDuckGo and returns top results as a formatted text.
    """
    searcher = DuckDuckGoSearcher(max_results=5)
    search_results = searcher.search(query)
    if not search_results:
        return "No results found."
    output_lines = []
    for r in search_results:
        line = f"- {r['title']}: {r['url']}\n  » {r['snippet']}"
        output_lines.append(line)

    return "\n".join(output_lines)


@tool
def get_current_datetime():
    """Returns the current date and time as a string."""
    now = datetime.now()
    print("Current date and time:", now)
    return now.strftime("%d-%m-%Y %H:%M:%S")


tools = [duckduckgo_search, get_current_datetime]


def safe_load_silero_vad():
    """
    Loads and returns the Silero VAD model (and utils).
    Raises an exception if loading fails.
    """
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        vad_model.eval()
        print("Silero VAD model loaded successfully.")
        return vad_model, utils
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not load Silero VAD model: {e}")


def safe_load_nemo_asr(model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
    """
    Loads and returns the NeMo ASR model.
    Raises an exception if loading fails.
    """
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name)
        print("NeMo ASR model loaded successfully.")
        return asr_model
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not load NeMo ASR model: {e}")


vad_model, vad_utils = safe_load_silero_vad()
(get_speech_timestamps, save_audio, read_audio,
 VADIterator, collect_chunks) = vad_utils  # if needed
asr_model = safe_load_nemo_asr()


def is_microphone_available() -> bool:
    """
    Checks if the system microphone is available and configured for the desired
    sample rate and number of channels.
    """
    try:
        sd.check_input_settings(
            samplerate=RATE, channels=CHANNELS, dtype='int16')
        print("Microphone detected and configured.")
        return True
    except Exception as e:
        print(f"[Warning] Microphone not available/config issue: {e}. "
              "Falling back to text input.")
    return False


mic_available = is_microphone_available()


def listen_and_transcribe(max_silence_ms: int = 300, vad_threshold: float = 0.5):
    """
    Listen from the microphone and transcribe segments of speech.

    - max_silence_ms: Maximum silence allowed before concluding the utterance.
    - vad_threshold: Probability threshold for Silero VAD to detect speech.
    Yields each transcribed utterance as a string.
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

    print("Listening...")

    try:
        while True:
            frame_data, overflow = stream.read(FRAME_SIZE)
            if overflow:
                print("[Warning] Input overflow detected.")

            frame_f32 = frame_data.flatten().astype(np.float32) / 32768.0
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


memory = InMemoryStore()
model = ChatterboxTTS.from_pretrained(device="cuda")


def generate_speech(
    model: ChatterboxTTS,
    text: str,
    audio_prompt_path: str = AUDIO_PROMPT_PATH,
    exaggeration: float = EXAGGERATION_FACTOR,
    temperature: float = TEMPERATURE_VALUE,
    cfgw: float = CFG_WEIGHT_VALUE,
) -> np.ndarray:
    """
    Generate audio from text using the ChatterboxTTS model; returns a NumPy waveform.
    """
    wav_tensor = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    # Remove batch dim, move to CPU, convert to NumPy
    return wav_tensor.squeeze(0).cpu().numpy()


def play_audio(numpy_wav: np.ndarray, sample_rate: int):
    """
    Plays a NumPy waveform at the specified sample rate using sounddevice.
    """
    print(f"Playing audio at {sample_rate} Hz...")
    sd.play(numpy_wav, samplerate=sample_rate)
    sd.wait()  # Block until playback finishes
    print("Playback finished.")
    return sample_rate, numpy_wav


class Response(BaseModel):
    """
    Defines a schema for the LLM's response if you want to parse it with LangChain's
    PydanticOutputParser.
    """
    action: str = "A brief stage direction or non-verbal action."
    dialogue: str = "The waifu's spoken response to the user."


parser = PydanticOutputParser(pydantic_object=Response)
llm = init_chat_model("gpt-4.1-nano", model_provider="openai")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{personality}"),
        ("human", "{query}"),
    ]
)
llm_with_tools = llm.bind_tools(tools)
chain = prompt | llm_with_tools
config = {"configurable": {"thread_id": "1"}}


def main():
    print("Rin is waiting for you.")
    while True:
        bench1 = timer()
        for input_text in listen_and_transcribe():
            print("You said:", input_text)
            bench2 = timer()
            # Invoke the chain with the user input and personality
            response = chain.invoke(
                {"query": input_text, "personality": personality},
                config
            )
            response_content = str(response.content)
            print("Rin says:", response_content)

            bench3 = timer()

            # Generate and play TTS audio
            wav_data = generate_speech(model, response_content)
            bench4 = timer()
            play_audio(wav_data, model.sr)

            # Print some timing info
            print(
                f"Benchmarks:\n"
                f"  - Input Transcription: {bench2 - bench1:.2f}s\n"
                f"  - LLM Response:        {bench3 - bench2:.2f}s\n"
                f"  - TTS Generation:      {bench4 - bench3:.2f}s\n"
            )
            # Break after handling a single utterance; then loop to await next input
            break


if __name__ == "__main__":
    main()
