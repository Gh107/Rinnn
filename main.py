from ai_tools import *
from personality import *
from audio import *

from dotenv import load_dotenv
from chatterbox.tts import ChatterboxTTS
import sounddevice as sd
from timeit import default_timer as timer

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()
memory = InMemoryStore()
AUDIO_PROMPT_PATH = "vc.mp3"

model = ChatterboxTTS.from_pretrained(device="cuda")


def generate(model, text, audio_prompt_path, exaggeration, temperature, cfgw):
    """
    Generates audio from text using the ChatterboxTTS model, plays it,
    and returns the sample rate and waveform.
    """
    # Generate audio waveform as a PyTorch tensor
    # This tensor might be on the GPU if 'model' was loaded to CUDA
    wav_tensor = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )

    # Get the sample rate from the model
    sample_rate = model.sr

    # Prepare the waveform for sounddevice:
    # 1. Remove batch dimension (ChatterboxTTS output is typically [1, N_samples]).
    # 2. Move tensor to CPU (if it was on CUDA, as 'sounddevice' needs CPU data).
    # 3. Convert to a NumPy array.
    return wav_tensor.squeeze(0).cpu().numpy()


def play_audio(numpy_wav, sample_rate):
    # Play the audio using sounddevice
    # sounddevice expects data in a format like float32 in the range [-1.0, 1.0],
    # which ChatterboxTTS typically provides.
    print(f"Playing audio at {sample_rate} Hz...")
    sd.play(numpy_wav, samplerate=sample_rate)

    # Wait for the playback to complete before proceeding
    sd.wait()
    print("Playback finished.")

    # As in the original function's intent, return the sample rate and the NumPy waveform.
    # This allows the caller to potentially save the file or perform other operations if needed.
    return (sample_rate, numpy_wav)


class Response(BaseModel):
    "Response to the roleplay prompt"
    action: str = "A brief stage direction or non-verbal action."
    dialogue: str = "The waifu's spoken response to the user."


parser = PydanticOutputParser(pydantic_object=Response)
llm = init_chat_model("gpt-4.1-nano", model_provider="openai")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{personality}",
        ),
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
        for input_text in listen_and_transcribe():  # Assuming listen_and_transcribe is defined
            print("You said:", input_text)
            bench2 = timer()
            response = chain.invoke(
                # Ensure 'personality' is defined
                {"query": input_text, "personality": personality},
                config
            )
            response_content = str(response.content)
            # Changed from print(response_content) for clarity
            print("Rin says:", response_content)
            bench3 = timer()

            # Define parameters for TTS generation
            # You can adjust these values as needed
            exaggeration_factor = 1.0  # Default: 1.0
            temperature_value = 0.7    # Default: 0.7
            # Default: 3.0 (sometimes referred to as guidance)
            cfg_weight_value = 3.0

            # Call the corrected generate function to synthesize and play audio
            wav_data = generate(
                model,
                response_content,
                AUDIO_PROMPT_PATH,
                exaggeration_factor,
                temperature_value,
                cfg_weight_value
            )
            bench4 = timer()
            _, _ = play_audio(wav_data, model.sr)
            # The 'break' was in your original loop; keeping it if it's intentional
            # This break will cause the 'listen_and_transcribe' loop to exit after one iteration
            # and the outer 'while True' loop to restart, waiting for new input.
            print(
                f"Benchmarks: Input: {bench2 - bench1:.2f}s, Response: {bench3 - bench2:.2f}s, TTS: {bench4 - bench3:.2f}s")
            break


if __name__ == "__main__":
    main()
