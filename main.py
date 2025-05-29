from ai_tools import *
from personality import *
from audio import *

from dotenv import load_dotenv
from kokoro import KPipeline
import sounddevice as sd

from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()
memory = InMemoryStore()

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
        # Use voice input
        for input_text in listen_and_transcribe():
            print("You said:", input_text)
            if input_text.lower() == "exit":
                return
            response = chain.invoke(
                {"query": input_text, "personality": personality},
                config
            )
            response_content = str(response.content)
            print(response_content)
            synthesize_and_play(response_content)
            break


if __name__ == "__main__":
    main()
