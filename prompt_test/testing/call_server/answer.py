import pyaudio
import wave
import tempfile
from faster_whisper import WhisperModel

# Function to record audio
def record_audio():
    # Configuration for recording
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
        wav_path = temp_wav_file.name
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    return wav_path

# Function to transcribe audio using Faster Whisper
def transcribe_audio(wav_path):
    # Load the Whisper model
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Transcribe the recorded audio
    segments, info = model.transcribe(wav_path, beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    transcription = ""
    for segment in segments:
        transcription += segment.text + " "

    return transcription.strip()

# Main execution block
if __name__ == "__main__":
    # Record audio
    wav_path = record_audio()

    # Transcribe the recorded audio
    transcription = transcribe_audio(wav_path)

    # Print the transcribed text
    print("Transcription:", transcription)



