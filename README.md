# Enhanced Real-Time Medical Conversation Transcription

**Real-time Whisper** is a high-accuracy, low-latency system for streaming transcription of medical conversations.
It combines OpenAI Whisper Large-v3 for speech recognition with NVIDIA NeMo (and optional Pyannote) for speaker diarization, optimized for real-time performance.

---

## 🚀 Features

- **Real-time audio capture** using PyAudio with optimized buffer management.
- **Fast streaming transcription** powered by Whisper large-v3 with timestamp support.
- **Advanced speaker diarization** via NVIDIA NeMo or Pyannote fallback with optimized speaker thresholds.
- **Modern Material UI interface** with intuitive controls and professional design.
- **Configurable speaker count** to optimize diarization for your specific scenario.
- **Color-coded transcription display** for easy speaker identification.
- **Multi-threaded pipeline** for VAD, diarization, transcription, and output.
- **GPU acceleration** (CUDA/MPS) to maximize throughput.
- **High-quality Voice Activity Detection (VAD)** using Silero VAD with tuned thresholds for optimized real-time performance.
- **Low-latency configuration** with optimized buffer sizes and speech segmentation parameters.
- **Anti-hallucination measures** to improve transcription quality in noisy environments.
- **Configurable parameters** including language, speaker count, chunk size, and silence thresholds.

---

## 🛠️ Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm for the React client
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

2. **Create a virtual environment and install backend dependencies**
   ```bash
   python3 -m venv whisperenv
   source whisperenv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install client dependencies**
   ```bash
   cd client
   npm install
   cd ..
   ```

---

## 🚀 Running the Application

### Starting the Backend Server

```bash
# Activate the virtual environment if not already activated
source whisperenv/bin/activate

# Start the server
python app.py
```

For macOS users with Apple Silicon (M1/M2/M3), use the MPS fallback option for better performance:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1 && uvicorn app:app --host 0.0.0.0 --port 8000
```

Alternatively, use the provided script:
```bash
./start_server.sh
```

### Starting the React Client

```bash
cd client
npm start
```

Alternatively, use the provided script:
```bash
./start_client.sh
```

### Using the Application

1. Open your browser and navigate to `http://localhost:3000`
2. Configure your settings:
   - Select your preferred language from the dropdown
   - Set the number of speakers you expect in the conversation (1-6)
3. Click "Start Transcription" to begin
4. Speak into your microphone
5. View real-time transcriptions in the color-coded display
6. Click "Stop Transcription" when finished

---

## 🐳 Docker Deployment

### Using Docker Compose

1. Make sure Docker and Docker Compose are installed
2. Build and start the services:
   ```bash
   docker-compose up -d
   ```
3. Access the application at `http://localhost:3000` (client) and `http://localhost:8080` (API)

### Building and Running the Docker Image Directly

```bash
# Build the image
docker build -t real-time-whisper .

# Run the container
docker run -p 8080:80 -e HF_TOKEN=your_token real-time-whisper
```

---

## 🚀 GitHub to RunPod Deployment

### 1. Set up your GitHub Repository

1. **Create a new GitHub repository**:
   - Go to github.com and create a new repository
   - Name it something like "real-time-whisper"

2. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/real-time-whisper.git
   git push -u origin main
   ```

3. **Make sure the following files are included**:
   - `Dockerfile`: Contains environment setup
   - `requirements.txt`: Lists all Python dependencies
   - `app.py`: Main FastAPI application
   - `real_time_diarization.py`: Contains the transcription logic
   - `client/`: React client application

### 2. Create a RunPod Pod from GitHub

1. **Log into RunPod**:
   - Go to https://www.runpod.io/ and log in

2. **Create a new pod**:
   - Click on "Secure Cloud" > "Deploy"
   - Select a GPU template (at least 16GB VRAM recommended)
   - Under "Deploy Options", select "GitHub Repository"
   - Enter your repository URL
   - Set "Container HTTP Port" to 80
   - Add environment variables:
     - Key: `HF_TOKEN`, Value: your Hugging Face token (mark as secret)

3. **Configure Pod settings**:
   - Select a volume size (at least 10GB recommended)
   - Choose "Expose HTTP port" to enable API access

4. **Deploy**:
   - Click "Deploy" and wait for your pod to start
   - RunPod will automatically clone your repository and build using your Dockerfile

### 3. Access Your API

Once deployed, your API will be available at:
- API: `https://[pod-id]-80.proxy.runpod.net/`
- WebSocket endpoint: `wss://[pod-id]-80.proxy.runpod.net/ws`
- Health check: `https://[pod-id]-80.proxy.runpod.net/health`

---

## 🐞 Troubleshooting

- **Permission errors** on macOS: grant microphone access in System Preferences.
- **PortAudio errors**: ensure `portaudio` is installed and headers are available.
- **Model loading issues**: check your network and Hugging Face credentials.
- **Pyannote fallback** requires a valid `HF_TOKEN` in `.env`.
- **React client issues**: ensure you have installed all npm dependencies with `npm install`.
- **Material UI not working**: check that all MUI dependencies were installed correctly.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Please open an issue or submit a pull request on GitHub. 