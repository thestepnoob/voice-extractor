# Podcast Multi-Speaker Batch Editor

A powerful local audio processing tool designed for high-quality vocal separation and speaker diarization. This project provides a GUI to separate vocals from music/noise and split speech into individual speaker tracks - perfect for podcast editing.

## 🎙️ Features

- **Vocal Separation**: Uses **Meta's Demucs** (htdemucs) to extract high-quality vocals from any audio file.
- **Speaker Diarization**: Leverages **pyannote.audio** to identify different speakers and detect overlaps.
- **Batch Processing**: Queue multiple files for automated processing.
- **Interactive Review**: A built-in web-based GUI (built with **NiceGUI**) to review diarization results and export corrected tracks.
- **Memory Efficient**: Implements chunked processing to handle large audio files even on consumer-grade hardware.

## 🛠 Tech Stack

- **GUI**: NiceGUI (Web-based)
- **Audio Processing**: Demucs, Pyannote.audio, Pydub, Torchaudio
- **Deep Learning**: PyTorch (CUDA acceleration supported)
- **Data & Viz**: Matplotlib, Numpy, Pandas

## 📋 Setup & Installation

### 1. Prerequisites

- **FFmpeg**: Essential for audio conversion (pydub & demucs).
    - **Windows**: Install via `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
- **Python**: 3.10+
- **CUDA**: Highly recommended for faster processing (GPU).
- **Hugging Face Token**: Required for downloading the pyannote models. Accept the terms on Hugging Face for the following models:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 2. Installation

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Usage

1. Start the application:

   ```bash
   python voice_extractor.py
   ```

2. Open your browser to the URL shown in the terminal (usually `http://localhost:8080`).
3. Enter your Hugging Face token and upload your podcast files.

## 📜 License

MIT License.
