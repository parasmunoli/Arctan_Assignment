import numpy as np
import pyaudio
import wave
from dataclasses import dataclass
from typing import Optional, List, Any
import queue

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters"""
    chunkSize: int = 4096
    sampleRate: int = 16000
    channels: int = 1
    format: int = pyaudio.paFloat32
    deviceIndex: Optional[int] = None

class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.isRecording = False
        self.audioQueue = queue.Queue()

    def startStream(self):
        """Start the audio stream"""
        if self.stream is not None:
            return

        def callback(in_data: bytes, *_: Any) -> tuple[bytes, int]:
            """
            Audio stream callback function
            Args:
                in_data: Input audio data
                *_: Unused callback parameters (frame_count, time_info, status)
            Returns:
                Tuple of (audio_data, stream_status)
            """
            if self.isRecording:
                audioData = np.frombuffer(in_data, dtype=np.float32)
                self.audioQueue.put(audioData)
            return (in_data, pyaudio.paContinue)

        self.stream = self.audio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sampleRate,
            input=True,
            frames_per_buffer=self.config.chunkSize,
            stream_callback=callback
        )

        self.isRecording = True
        self.stream.start_stream()

    def getLatestChunk(self) -> Optional[np.ndarray]:
        """Get the latest audio chunk from the queue"""
        try:
            return self.audioQueue.get_nowait()
        except queue.Empty:
            return None

    def stopStream(self):
        """Stop the audio stream"""
        self.isRecording = False
        if self.stream is not None:
            self.stream.stopStream()
            self.stream.close()
            self.stream = None

    def saveAudio(self, filename: str, audio_buffer: List[np.ndarray]):
        """Save recorded audio to a WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.config.format))
            wf.setframerate(self.config.sampleRate)

            for chunk in audio_buffer:
                wf.writeframes(chunk.tobytes())

    def __del__(self):
        """Cleanup resources"""
        self.stopStream()
        self.audio.terminate()