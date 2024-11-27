import os
import wave
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Define constants
NEON_GREEN = '\033[32m'
RESET_COLOR = '\033[0m'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define a set of phrases to ignore
IGNORE_PHRASES = {"thanks for watching", "thank you", "have a nice day"}

# Function to check if audio contains speech
def is_speech(audio_data, threshold=500):
    """Check if the audio data contains speech based on a simple volume threshold."""
    audio_array = np.frombuffer(audio_data, np.int16)
    return np.max(np.abs(audio_array)) > threshold

# Function to record an audio chunk
def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        if is_speech(data):
            frames.append(data)

    if frames:  # Only save if there is recorded audio
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7, language='en')
    transcription = ''.join(segment.text for segment in segments).strip()
    return transcription

def main():
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open recording stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    accumulated_transcription = ""

    try:
        while True:
            # Record an audio chunk
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)

            # Only transcribe if something was recorded
            if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
                # Transcribe the audio chunk
                transcription = transcribe_chunk(model, chunk_file)

                # Filter out transcriptions that are too short or irrelevant
                if (len(transcription) > 2 
                        and not any(phrase in transcription.lower() for phrase in IGNORE_PHRASES)):
                    print(NEON_GREEN + transcription + RESET_COLOR)
                    accumulated_transcription += transcription + " "

                # Delete the temporary file
                os.remove(chunk_file)

    except KeyboardInterrupt:
        print("Stopping...")

        # Write the accumulated transcription to a log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription.strip())

    finally:
        print("LOG: " + accumulated_transcription.strip())

        # Stop the recording stream
        stream.stop_stream()
        stream.close()

        # Terminate PyAudio
        p.terminate()

if __name__ == "__main__":
    main()


