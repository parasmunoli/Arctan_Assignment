import pyaudio
import wave

class AudioProcessor:
    def __init__(self, chunk_size=2048, sample_rate=16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.format = pyaudio.paInt16
        self.channels = 1
        self.audio_interface = pyaudio.PyAudio()
