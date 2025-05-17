from ai_tools import *
from personality import *
from audio import *

from dotenv import load_dotenv
import io
import sounddevice as sd
from scipy.io import wavfile

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# from indextts.infer import IndexTTS


load_dotenv()
memory = MemorySaver()
'''
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
voice = "vc.wav"


def synthesize_and_play(tts: IndexTTS, voice_prompt: str, text: str):
    """
    Generate speech for `text` using `voice_prompt`, return immediately and play through speakers.

    Args:
        tts: Initialized IndexTTS instance.
        voice_prompt: Path to the cloned voice WAV file.
        text: The text to synthesize.
    """
    # In-memory buffer
    buf = io.BytesIO()

    # Generate WAV into buffer
    tts.infer(voice_prompt, text, buf)
    buf.seek(0)

    # Read buffer
    sr, audio = wavfile.read(buf)

    # Play via sounddevice
    sd.play(audio, samplerate=sr)
    sd.wait()
'''

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

system_prompt = SystemMessage(
    content=personality)

agent_executor = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
    response_format=json_schema,
    checkpointer=memory
)

config = {"configurable": {"thread_id": "1"}}


def main():
    print("Rin is waiting for you.")
    while True:
        if mic_available:
            # Use voice input
            for input_text in listen_and_transcribe():
                print("You said:", input_text)
                if input_text.lower() == "exit":
                    return
                response = agent_executor.invoke(
                    {"messages": [HumanMessage(content=input_text)]},
                    config
                )
                print("Mood:", response["structured_response"].get("mood"))
                print("Action:", response["structured_response"].get("action"))
                print("Dialogue:",
                      response["structured_response"].get("dialogue"))
                # synthesize_and_play(
                #    tts, voice, response["structured_response"].get("dialogue"))
                break
        else:
            # Fallback to text input
            input_text = input("User: ")
            if input_text.lower() == "exit":
                break
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=input_text)]},
                config
            )
            print("Mood:", response["structured_response"].get("mood"))
            print("Action:", response["structured_response"].get("action"))
            print("Dialogue:",
                  response["structured_response"].get("dialogue"))
            # synthesize_and_play(
            #    tts, voice, response["structured_response"].get("dialogue"))


if __name__ == "__main__":
    main()
