from datetime import datetime
from timeit import default_timer as timer
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import nemo.collections.asr as nemo_asr
import chromadb
from chromadb.utils import embedding_functions
import uuid
import json

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

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

# ChromaDB settings
CHROMA_PERSIST_PATH = "./chroma_db"
COLLECTION_NAME = "conversation_history"


class ConversationMemory:
    """
    Manages conversation history using ChromaDB with MiniLM embeddings.
    """

    def __init__(self, persist_directory: str = CHROMA_PERSIST_PATH):
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use MiniLM embedding function from sentence-transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"ChromaDB initialized with {self.collection.count()} existing conversations.")

    def add_conversation(self, user_input: str, assistant_response: str, summary: str = None):
        """
        Add a conversation to the database.

        :param user_input: The user's input text
        :param assistant_response: The assistant's response
        :param summary: Optional summary of the conversation
        """
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Create the full conversation text for embedding
        full_text = f"User: {user_input}\nAssistant: {assistant_response}"

        # If no summary provided, use the full conversation as document
        document = summary if summary else full_text

        metadata = {
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": timestamp,
            "has_summary": bool(summary)
        }

        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[conversation_id]
        )

        print(f"Conversation added to database (ID: {conversation_id[:8]}...)")

    def retrieve_relevant_conversations(self, query: str, n_results: int = 2) -> list[dict]:
        """
        Retrieve the most relevant conversations based on the query.

        :param query: The search query (typically the user's current input)
        :param n_results: Number of results to return
        :return: List of relevant conversations with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        conversations = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                conv = {
                    "user_input": metadata.get("user_input", ""),
                    "assistant_response": metadata.get("assistant_response", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "relevance_score": 1 - results['distances'][0][i] if results['distances'] else 0
                }
                conversations.append(conv)

        return conversations

    def format_context_for_prompt(self, conversations: list[dict]) -> str:
        """
        Format retrieved conversations for inclusion in the system prompt.

        :param conversations: List of conversation dictionaries
        :return: Formatted string for the prompt
        """
        if not conversations:
            return ""

        context_parts = ["Previous relevant conversations:"]
        for i, conv in enumerate(conversations, 1):
            context_parts.append(
                f"\n[Conversation {i} - {conv['timestamp'][:10]}]\n"
                f"User: {conv['user_input']}\n"
                f"You: {conv['assistant_response']}"
            )

        return "\n".join(context_parts)


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

# Create a tool mapping for easy access
tool_mapping = {tool.name: tool for tool in tools}


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

# Initialize conversation memory
conversation_memory = ConversationMemory()


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
    Defines a schema for the LLM's response.
    """
    action: str = Field(
        description="A brief stage direction or non-verbal action, e.g., *smiles warmly*, *tilts head curiously*")
    dialogue: str = Field(
        description="The character's spoken response to the user")


def summarize_conversation(llm, user_input: str, assistant_response: str) -> str:
    """
    Generate a concise summary of a conversation exchange.
    """
    summary_prompt = f"""Summarize this conversation exchange in 1-2 sentences:
User: {user_input}
Assistant: {assistant_response}

Summary:"""

    summary_response = llm.invoke(summary_prompt)
    return summary_response.content.strip()


def parse_response(response_content: str, parser: PydanticOutputParser) -> Response:
    """
    Parse the response content into a Response object.
    """
    try:
        # Try to parse as JSON first
        if isinstance(response_content, str):
            # Attempt to extract JSON from the content
            if '{' in response_content and '}' in response_content:
                # Find the JSON portion
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}') + 1
                json_str = response_content[start_idx:end_idx]

                try:
                    parsed_dict = json.loads(json_str)
                    return Response(**parsed_dict)
                except json.JSONDecodeError:
                    pass

            # If JSON parsing fails, try the Pydantic parser
            return parser.parse(response_content)
    except Exception as e:
        print(f"Failed to parse structured response: {e}")
        # Fallback: return the raw content as dialogue
        return Response(
            action="*responds*",
            dialogue=str(response_content)
        )


def execute_tool_calls(tool_calls: list, tool_mapping: dict) -> list[ToolMessage]:
    """
    Execute tool calls and return ToolMessages with results.
    """
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call.get('id', str(uuid.uuid4()))

        print(f"Executing tool: {tool_name}")

        try:
            if tool_name in tool_mapping:
                tool_func = tool_mapping[tool_name]
                result = tool_func.invoke(tool_args)
            else:
                result = f"Error: Unknown tool '{tool_name}'"

            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            ))
        except Exception as e:
            print(f"Error executing tool {tool_name}: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error executing tool: {str(e)}",
                tool_call_id=tool_id
            ))

    return tool_messages


parser = PydanticOutputParser(pydantic_object=Response)
llm = init_chat_model("gpt-4.1-nano", model_provider="openai")

# Enhanced prompt template with format instructions
prompt_template = """{personality}

{context}

IMPORTANT: When responding to the user (and not using tools), you MUST format your response as a JSON object with the following structure:
{format_instructions}

Example response:
{{"action": "*smiles warmly*", "dialogue": "Oh, hello there! It's nice to see you again."}}

Remember to always include both an action (a brief stage direction) and dialogue (what you actually say) in your response."""

llm_with_tools = llm.bind_tools(tools)


def get_final_response(messages: list, context: str, format_instructions: str, config: dict):
    """
    Get the final response from the LLM, handling tool calls if necessary.
    Returns the final parsed Response object.
    """
    # Create the prompt with the system message
    system_prompt = prompt_template.format(
        personality=personality,
        context=context,
        format_instructions=format_instructions
    )

    # Build the full message list
    full_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages

    # Call the LLM
    response = llm_with_tools.invoke(full_messages, config)

    # Check if there are tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("Tool calls detected, executing...")

        # Execute the tool calls
        tool_messages = execute_tool_calls(response.tool_calls, tool_mapping)

        # Add the assistant's message with tool calls and tool results to messages
        messages.append(
            {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
        for tool_msg in tool_messages:
            messages.append({"role": "tool", "content": tool_msg.content,
                            "tool_call_id": tool_msg.tool_call_id})

        # Call the LLM again with the tool results
        return get_final_response(messages, context, format_instructions, config)

    # No tool calls, parse and return the response
    return parse_response(response.content, parser)


def main():
    print("Rin is waiting for you.")
    config = {"configurable": {"thread_id": "1"}}

    while True:
        bench1 = timer()
        for input_text in listen_and_transcribe():
            print("You said:", input_text)
            bench2 = timer()

            # Time the database query
            db_query_start = timer()
            # Retrieve relevant past conversations
            relevant_conversations = conversation_memory.retrieve_relevant_conversations(
                input_text, n_results=2
            )
            context = conversation_memory.format_context_for_prompt(
                relevant_conversations)
            db_query_end = timer()

            if context:
                print("Found relevant context from past conversations.")

            # Get format instructions from parser
            format_instructions = parser.get_format_instructions()

            # Get the final response (handling tool calls if necessary)
            messages = [{"role": "user", "content": input_text}]
            parsed_response = get_final_response(
                messages, context, format_instructions, config)

            # Display the parsed response
            print(f"[{parsed_response.action}]")
            print(f"Rin says: {parsed_response.dialogue}")

            bench3 = timer()

            # Generate and play TTS audio using only the dialogue
            wav_data = generate_speech(model, parsed_response.dialogue)
            bench4 = timer()
            play_audio(wav_data, model.sr)

            # Generate summary and store conversation
            bench5 = timer()
            # Store the full response (action + dialogue) for context
            full_response = f"{parsed_response.action} {parsed_response.dialogue}"
            summary = summarize_conversation(llm, input_text, full_response)
            conversation_memory.add_conversation(
                user_input=input_text,
                assistant_response=full_response,
                summary=summary
            )
            bench6 = timer()

            # Print timing info
            print(
                f"Benchmarks:\n"
                f"  - Input Transcription:  {bench2 - bench1:.2f}s\n"
                f"  - Database Query:       {db_query_end - db_query_start:.2f}s\n"
                f"  - LLM Response:         {bench3 - bench2:.2f}s\n"
                f"  - TTS Generation:       {bench4 - bench3:.2f}s\n"
                f"  - Memory Storage:       {bench6 - bench5:.2f}s\n"
            )
            # Break after handling a single utterance; then loop to await next input
            break


if __name__ == "__main__":
    main()
