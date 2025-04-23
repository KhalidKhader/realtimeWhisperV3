# Enhanced Real-Time Medical Conversation Transcription

**Real-time Whisper** is a high-accuracy, low-latency system for streaming transcription of medical conversations.
It combines OpenAI Whisper v3 Large for speech recognition with NVIDIA NeMo (and optional Pyannote) for speaker diarization.

---

## 🚀 Features

- **Real-time audio capture** using PyAudio with optimized buffer management.
- **Streaming transcription** powered by Whisper large-v3 with timestamp support.
- **Advanced speaker diarization** via NVIDIA NeMo or Pyannote fallback.
- **Doctor/patient role identification** using ML-based classification.
- **Multi-threaded pipeline** for VAD, diarization, transcription, and output.
- **GPU acceleration** (CUDA/MPS) to maximize throughput.
- **Ambient noise auto-calibration** at startup to adapt VAD thresholds to your room's noise floor
- **Configurable parameters** including chunk size, silence thresholds, and language.

---

## 🛠️ Prerequisites

- Python 3.8 or higher
- **PortAudio** (for PyAudio):
  - macOS: `brew install portaudio`
  - Linux: `sudo apt-get install portaudio19-dev`
- A microphone or audio input device.
- (Optional) **Hugging Face API token** in `.env` for Pyannote fallback:
  ```bash
  echo "HF_TOKEN=your_hf_api_token" > .env
  ```

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/real-time-whisper.git
   cd real-time-whisper
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv whisperenv
   source whisperenv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install numpy pyaudio soundfile python-dotenv scikit-learn transformers nemo_toolkit[asr] pyannote.audio
   ```

> Alternatively, if a `requirements.txt` is provided:
> ```bash
> pip install -r requirements.txt
> ```

---

## ⚙️ Configuration

The script can be customized in the `__main__` block of `rt.py`:

```python
config = {
    "language": "en",         # ISO code or Whisper auto-detect
    "chunk_size": 4000,        # Samples per buffer (e.g., 250ms)
    "sample_rate": 16000,      # Audio sampling rate
    "use_cuda": True,          # GPU/MPS acceleration
    "num_threads": 4,          # Number of threads for processing
    "silence_threshold": 0.01, # VAD energy threshold
    "min_voice_duration": 0.5, # Minimum speech segment (seconds)
    "min_silence_duration": 0.5, # Minimum silence to split segments
    # Ambient noise calibration parameters:
    "calibration_duration": 2.0,  # seconds to record ambient noise at startup
    "calibration_factor": 1.5,    # multiplier for ambient noise RMS to set silence_threshold
}
```

You can also override environment settings via a `.env` file if needed.

---

## ▶️ Usage

```bash
source whisperenv/bin/activate
python3 rt.py
```

- Press `Ctrl+C` to stop the transcription.
- Upon shutdown, a summary of session duration and detected speakers is printed.

---

## 🐞 Troubleshooting

- **Permission errors** on macOS: grant microphone access in System Preferences.
- **PortAudio errors**: ensure `portaudio` is installed and headers are available.
- **Model loading issues**: check your network and Hugging Face credentials.
- **Pyannote fallback** requires a valid `HF_TOKEN` in `.env`.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Please open an issue or submit a pull request on GitHub. 