import sounddevice as sd
import numpy as np
from pydub import AudioSegment

def record_audio(duration, filename):
    # Set the sample rate
    sample_rate = 44100  # Standard sample rate for audio

    # Record audio
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Convert to numpy array and save as WAV
    audio_data = audio_data.flatten()  # Flatten to 1D array
    audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=sample_rate, sample_width=2, channels=2)
    
    # Export as MP3
    audio_segment.export(filename, format="mp3")
    print(f"Audio saved as {filename}")

# Example usage
record_audio(duration=5, filename='output.mp3')  # Record for 5 seconds

