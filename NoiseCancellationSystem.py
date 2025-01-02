import numpy as np
import pyaudio
import wave
from scipy.signal import lfilter, butter, filtfilt
from collections import deque

# System Constants
CHUNK_SIZE = 2048  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit integer PCM)
CHANNELS = 1  # Mono audio input
RATE = 16000  # Sampling rate in Hz (16kHz is standard for speech)
LATENCY = 0.1  # Target latency in seconds
OUTPUT_FILE = "output.wav"  # Default output file name


class NoiseCancellationSystem:
    def __init__(self, operation_mode="single"):
        """
        Initialize the noise cancellation system with enhanced multi-speaker support.

        Parameters:
        operation_mode (str): Either 'single' for one primary speaker or 'multiple' for preserving multiple voices
        """
        self.operation_mode = operation_mode
        self.audio_buffer = []
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None

        # Voice detection and processing parameters
        self.voice_freq_ranges = [
            (85, 255),  # Male voice fundamental frequency range
            (165, 255),  # Female voice fundamental frequency range
            (250, 400)  # Children's voice fundamental frequency range
        ]
        self.energy_threshold = 0.1  # Base energy threshold for voice detection
        self.history_size = 50  # Size of the sliding window for energy history
        self.energy_history = deque(maxlen=self.history_size)
        self.adaptive_threshold = 0.0  # Dynamic threshold that adapts to ambient noise

        # Noise reduction parameters
        self.noise_floor = 0.1  # Minimum noise level to consider
        self.noise_reduction_factor = 0.7  # Factor for noise reduction (0-1)

        # Initialize multi-band filters for different voice ranges
        self.voice_filters = [self._create_bandpass_filter(low, high) for low, high in self.voice_freq_ranges]

    @staticmethod
    def _create_bandpass_filter(lowcut, highcut):
        """
        Create a bandpass filter for a specific frequency range.

        Parameters:
        lowcut (float): Lower frequency cutoff in Hz
        highcut (float): Upper frequency cutoff in Hz

        Returns:
        tuple: Filter coefficients (b, a) for the bandpass filter
        """
        nyquist = RATE / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        return butter(4, [low, high], btype='band')

    @staticmethod
    def _calculate_frame_energy(data):
        """
        Calculate the energy content of an audio frame.

        Parameters:
        data (numpy.ndarray): Input audio frame

        Returns:
        float: Energy level of the frame
        """
        return np.mean(np.square(data.astype(float)))

    def _update_adaptive_threshold(self, energy):
        """
        Update the adaptive threshold based on recent frame energies.
        Uses a sliding window approach to maintain current noise floor estimates.

        Parameters:
        energy (float): Energy of the current frame
        """
        self.energy_history.append(energy)
        # Use a percentile-based threshold to better handle varying noise conditions
        self.adaptive_threshold = np.percentile(self.energy_history, 70)

    def _preserve_multiple_speakers(self, data):
        """
        Process audio to preserve multiple speakers while reducing environmental noise.
        Uses multiple bandpass filters to cover different voice frequency ranges.

        Parameters:
        data (numpy.ndarray): Input audio data

        Returns:
        numpy.ndarray: Processed audio with preserved voices and reduced noise
        """
        # Convert to float for processing
        float_data = data.astype(float)

        # Initialize array for combined voice bands
        combined_voices = np.zeros_like(float_data)

        # Process each voice frequency range
        for b, a in self.voice_filters:
            # Apply bandpass filter for current voice range
            voice_band = filtfilt(b, a, float_data)

            # Calculate energy in this band
            band_energy = self._calculate_frame_energy(voice_band)

            # Only add this band if it contains significant energy
            if band_energy > self.adaptive_threshold:
                combined_voices += voice_band

        # Extract the residual (non-voice) components
        background = float_data - combined_voices

        # Apply noise reduction to background
        reduced_background = background * self.noise_reduction_factor

        # Combine preserved voices with reduced background
        enhanced_signal = combined_voices + reduced_background

        # Normalize the output
        max_val = np.max(np.abs(enhanced_signal))
        if max_val > 0:
            enhanced_signal = enhanced_signal * 32767 / max_val

        return enhanced_signal

    def noise_cancellation_multiple_speakers(self, data):
        """
        Main processing function for multiple speaker scenario.
        Preserves multiple voices while reducing environmental noise.

        Parameters:
        data (numpy.ndarray): Input audio data

        Returns:
        numpy.ndarray: Processed audio data
        """
        # Calculate frame energy and update threshold
        frame_energy = self._calculate_frame_energy(data)
        self._update_adaptive_threshold(frame_energy)

        # Apply multi-speaker preservation
        enhanced_signal = self._preserve_multiple_speakers(data)

        # Convert back to 16-bit integer format
        return np.int16(enhanced_signal)

    def noise_cancellation_single_speaker(self, data):
        """
        Process audio data for single speaker scenario.
        Enhances the primary speaker while reducing other sounds.

        Parameters:
        data (numpy.ndarray): Input audio data

        Returns:
        numpy.ndarray: Processed audio data
        """
        # Convert to float for processing
        float_data = data.astype(float)

        # Apply high-pass filter to remove low frequency noise
        filtered_data = self.highpass_filter(float_data)

        # Apply voice enhancement
        enhanced_data = self._apply_voice_enhancement(filtered_data)

        # Normalize the output
        max_val = np.max(np.abs(enhanced_data))
        if max_val > 0:
            enhanced_data = enhanced_data * 32767 / max_val

        return np.int16(enhanced_data)

    def _apply_voice_enhancement(self, data):
        """
        Apply voice enhancement for single speaker mode.

        Parameters:
        data (numpy.ndarray): Input audio data

        Returns:
        numpy.ndarray: Enhanced audio data
        """
        # Use the first voice filter (typical speaking range)
        b, a = self.voice_filters[0]
        voice_band = filtfilt(b, a, data)

        # Calculate frame energy
        frame_energy = self._calculate_frame_energy(data)
        self._update_adaptive_threshold(frame_energy)

        if frame_energy > self.adaptive_threshold:
            # Enhance voice band
            enhancement_factor = 1.5
            voice_band *= enhancement_factor

            # Reduce background
            background = data - voice_band
            background *= self.noise_reduction_factor

            return voice_band + background
        else:
            # Reduce non-voice frames
            return data * 0.5

    @staticmethod
    def highpass_filter(data):
        """
        Apply a high-pass filter to remove low-frequency noise.

        Parameters:
        data (numpy.ndarray): Input audio data

        Returns:
        numpy.ndarray: Filtered audio data
        """
        b = [0.5, -0.5]  # Filter coefficients (numerator)
        a = [1, -0.9]  # Filter coefficients (denominator)
        return lfilter(b, a, data)

    def audio_callback(self, in_data, *_):
        """
        Real-time audio processing callback function.

        Parameters:
        in_data (bytes): Raw audio data from the input stream

        Returns:
        tuple: Processed audio data and PyAudio flow control flag
        """
        data = np.frombuffer(in_data, dtype=np.int16)

        if self.operation_mode == "single":
            processed_data = self.noise_cancellation_single_speaker(data)
        else:
            processed_data = self.noise_cancellation_multiple_speakers(data)

        self.audio_buffer.append(processed_data)
        return processed_data.tobytes(), pyaudio.paContinue

    def start_stream(self):
        """
        Initialize and start the audio stream.
        """
        self.stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def stop_stream(self):
        """
        Stop the audio stream and clean up resources.
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()

    def save_audio_to_file(self, filename):
        """
        Save the processed audio to a WAV file.

        Parameters:
        filename (str): Output file path
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.audio_buffer))

    def run(self):
        """
        Main execution method for the noise cancellation system.
        """
        print(f"Starting real-time noise cancellation in {self.operation_mode} speaker mode...")
        print("Press Ctrl+C to stop.")

        try:
            self.start_stream()
            while self.stream.is_active():
                pass
        except KeyboardInterrupt:
            print("\nStopping audio stream...")
        finally:
            self.stop_stream()
            print("Saving processed audio to file...")
            self.save_audio_to_file(OUTPUT_FILE)
            print(f"Audio saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    user_mode = input("Enter mode (single/multiple): ").strip().lower()
    if user_mode not in ["single", "multiple"]:
        print("Invalid mode. Defaulting to 'single'.")
        user_mode = "single"

    system = NoiseCancellationSystem(operation_mode=user_mode)
    system.run()