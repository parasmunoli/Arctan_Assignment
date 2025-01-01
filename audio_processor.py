import pyaudio
import wave

class AudioProcessor:
    def __init__(self, chunk_size=2048, sample_rate=16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.format = pyaudio.paInt16
        self.channels = 1
        self.audio_interface = pyaudio.PyAudio()

    def recordAudio(self, duration_seconds, output_file):
        """Capture audio from the microphone."""
        stream = self.audio_interface.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print(f"Recording for {duration_seconds} seconds...")
        frames = []

        for _ in range(0, int(self.sample_rate / self.chunk_size * duration_seconds)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        print("Recording complete.")
        stream.stop_stream()
        stream.close()

        # Save to WAV file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio_interface.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

        print(f"Audio saved to {output_file}")
