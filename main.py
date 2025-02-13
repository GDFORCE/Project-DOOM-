import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero
from livekit.agents.tts import FallbackAdapter, TTS, TTSCapabilities
import ollama
import pyttsx3
from vosk import Model, KaldiRecognizer
import pyaudio

load_dotenv()

# Initialize the Ollama LLM
ollama_model = 'llama3.2'  # Replace with your desired model
ollama.pull(ollama_model)

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Initialize Vosk STT model
stt_model = Model("C:\\Users\\hp\\Downloads\\vosk-model-small-en-us-0.15\\vosk-model-small-en-us-0.15")  # Download and specify the path to the Vosk model

class OllamaLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate_text(self, prompt: str):
        response = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
        return response.message.content

class VoskSTT:
    def __init__(self, model):
        self.model = model
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio = pyaudio.PyAudio()

    def transcribe(self):
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
        stream.start_stream()
        print("Listening...")
        while True:
            data = stream.read(4096)
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                text = result.get('text', '')
                if text:
                    return text

class Pyttsx3TTS(TTS):
    def __init__(self, engine):
        # Specify TTS capabilities: streaming set to False
        super().__init__(capabilities=TTSCapabilities(streaming=False), sample_rate=16000, num_channels=1)
        self.engine = engine

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, avoiding the usage of unpronounceable punctuation."
        ),
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Use the FallbackAdapter to handle TTS with capabilities
    tts_adapter = FallbackAdapter([Pyttsx3TTS(tts_engine)])

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=VoskSTT(stt_model),
        llm=OllamaLLM(model_name=ollama_model),
        tts=tts_adapter,
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today!", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
