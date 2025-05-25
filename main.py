from ai_tools import *
from personality import *
from audio import *

from dotenv import load_dotenv
from kokoro import KPipeline
import sounddevice as sd

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
memory = MemorySaver()

pipeline = KPipeline(lang_code="a")


def synthesize_and_play(text: str):
    """
    Generate speech for `text` using the predefined Kokoro pipeline, return immediately and play through speakers.

    Args:
        text: The text to synthesize.
    """
    # The pipeline directly returns the audio data as a torch.Tensor
    for _, (_, _, audio_tensor) in enumerate(pipeline(text, voice="af_nicole")):
        # Ensure the tensor is on CPU
        audio_tensor = audio_tensor.cpu()

        # Scale and convert to float32 (sounddevice typically expects float32 or int16)
        sd.play(audio_tensor, samplerate=24000)
        sd.wait()
        break  # We only need the first chunk for simple playback if text is short


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
                print("Action:", response["structured_response"].get("action"))
                print("Dialogue:",
                      response["structured_response"].get("dialogue"))
                synthesize_and_play(
                    response["structured_response"].get("dialogue"))
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
            print("Action:", response["structured_response"].get("action"))
            print("Dialogue:",
                  response["structured_response"].get("dialogue"))
            synthesize_and_play(
                response["structured_response"].get("dialogue"))


if __name__ == "__main__":
    main()
