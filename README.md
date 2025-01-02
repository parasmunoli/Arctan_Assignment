# Arctan Assignment: Real-Time Noise Cancellation System

Welcome to the **Arctan Assignment** repository! This project implements a **Real-Time Noise Cancellation System** that processes audio streams to enhance voice quality and reduce environmental noise. The system supports both single-speaker and multi-speaker modes and is designed for live audio processing.

---

## Features

- **Real-Time Processing**: Captures and processes audio in real-time using the default system microphone.
- **Single vs. Multi-Speaker Modes**:
  - *Single*: Enhances the primary speaker's voice.
  - *Multiple*: Preserves multiple voices while reducing noise.
- **Adaptive Noise Reduction**: Dynamically adjusts to ambient noise levels.
- **Voice Detection**: Identifies and enhances voice frequency ranges.
- **Output Normalization**: Ensures processed audio maintains a balanced amplitude.
- **Audio Export**: Saves processed audio as a `.wav` file.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.10+
- `pip` (Python package manager)

### Step 1: Clone the Repository

```bash
$ git clone https://github.com/parasmunoli/Arctan_Assignment.git
$ cd Arctan_Assignment
```

### Step 2: Install Dependencies

Use the `requirements.txt` file to install the necessary Python packages:

```bash
$ pip install -r requirements.txt
```

> **Note**: For PyAudio, additional setup may be required. Refer to [PyAudio Installation Guide](https://people.csail.mit.edu/hubert/pyaudio/#downloads) if issues occur.

---

## Usage

### Running the System

Run the Python script and choose the mode of operation (single or multiple speaker):

```bash
$ python NoiseCancellationSystem.py
```

### Input Prompt

You will be prompted to select the operation mode:

- **`Single`**: Optimized for one primary speaker.
- **`Multiple`**: Designed to preserve multiple voices while reducing noise.

If no valid mode is entered, the system defaults to `Single`.

### Real-Time Audio Processing

- The system uses the default system microphone for input and outputs the processed audio in real-time.
- Press `Ctrl+C` to stop the system.

### Output

The processed audio is saved as a `.wav` file:

```plaintext
output.wav
```

---

## Code Structure

- **`NoiseCancellationSystem.py`**: Main script containing the implementation of the real-time noise cancellation system.
- **`requirements.txt`**: List of dependencies for the project.

---

## Example Workflow

1. Start the script and select `single` or `multiple` mode.
2. Speak into your microphone or play a sound.
3. Stop the program by pressing `Ctrl+C`.
4. Find the processed audio saved in `output.wav`.

---

## Dependencies

The following Python libraries are required:

- `numpy`: For numerical computations.
- `pyaudio`: For real-time audio processing.
- `scipy`: For signal processing tasks.

Install all dependencies using the `requirements.txt` file:

```bash
$ pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add your message'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

---
