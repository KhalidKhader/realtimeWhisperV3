#!/usr/bin/env python3
"""
Enhanced Real-Time Medical Conversation Transcription
====================================================
Uses Whisper v3 Large with NVIDIA NeMo for diarization to provide high-accuracy,
low-latency real-time transcription of medical conversations.

Features:
- Real-time audio capture with optimized buffer management
- Streaming transcription with Whisper large-v3
- Advanced speaker diarization with NVIDIA NeMo
- Doctor/patient identification with ML-based classification
- Multi-threaded processing for improved responsiveness
- GPU acceleration for high performance
"""

import os
import sys
import json
import time
import logging
import queue
import threading
import numpy as np
import torch
import pyaudio
import soundfile as sf
import wave
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline as PyannotePipeline
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("realtime_transcribe")

# Constants
DEFAULT_LANGUAGE = "en"  # Default language (ISO code), can be overridden

class AudioBuffer:
    """Enhanced audio buffer with voice activity detection."""
    def __init__(self, sample_rate=16000, max_size_seconds=30):
        self.sample_rate = sample_rate
        self.max_size = max_size_seconds * sample_rate
        self.buffer = np.array([], dtype=np.float32)
        self.vad_buffer = []  # Tracks which chunks contain voice
        self.lock = threading.Lock()
    
    def add(self, audio_chunk, has_voice=True):
        """Add audio chunk to buffer with voice activity flag."""
        with self.lock:
            self.buffer = np.concatenate([self.buffer, audio_chunk])
            self.vad_buffer.extend([has_voice] * len(audio_chunk))
            
            # Trim if exceeding max size
            if len(self.buffer) > self.max_size:
                excess = len(self.buffer) - self.max_size
                self.buffer = self.buffer[excess:]
                self.vad_buffer = self.vad_buffer[excess:]
    
    def get(self, clear=True):
        """Get buffer contents and optionally clear."""
        with self.lock:
            buffer_copy = self.buffer.copy()
            if clear:
                self.buffer = np.array([], dtype=np.float32)
                self.vad_buffer = []
            return buffer_copy
    
    def get_voice_segments(self, clear=True):
        """Extract segments that contain voice activity."""
        with self.lock:
            # Find continuous voice segments
            voice_mask = np.array(self.vad_buffer, dtype=bool)
            if not any(voice_mask):
                return [] if clear else []
                
            # Get voice segments
            segments = []
            current_segment = []
            for i, has_voice in enumerate(voice_mask):
                if has_voice:
                    current_segment.append(i)
                elif current_segment:
                    segments.append((min(current_segment), max(current_segment)+1))
                    current_segment = []
            
            if current_segment:  # Add final segment
                segments.append((min(current_segment), max(current_segment)+1))
            
            # Extract audio segments
            audio_segments = []
            for start, end in segments:
                segment = self.buffer[start:end]
                if len(segment) > 0:
                    audio_segments.append(segment)
            
            if clear:
                self.buffer = np.array([], dtype=np.float32)
                self.vad_buffer = []
                
            return audio_segments
    
    def duration(self):
        """Get buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

class SpeakerProfile:
    """Speaker profile for improved diarization."""
    def __init__(self, id, embedding=None, role=None):
        self.id = id
        self.embedding = embedding
        self.role = role
        self.utterances = []
        self.total_duration = 0.0
    
    def update_embedding(self, new_embedding, duration):
        """Update embedding using weighted average based on duration."""
        if self.embedding is None:
            self.embedding = new_embedding
        else:
            # Weighted average by duration
            weight = duration / (self.total_duration + duration)
            self.embedding = (1-weight) * self.embedding + weight * new_embedding
        
        self.total_duration += duration

class RealTimeTranscriber:
    def __init__(self, config=None):
        # Default configuration thresholds tuned for clarity
        default_config = {
            "sample_rate": 16000,
            "chunk_size": 4000,        # 250ms chunks
            "buffer_size": 30,         # 30s context buffer
            "silence_threshold": 0.005, # VAD energy threshold (more sensitive)
            "calibration_duration": 2.0, # seconds to record ambient noise at startup
            "calibration_factor": 1.5,   # multiplier for ambient noise RMS to set silence_threshold
            "min_voice_duration": 0.2,   # allow shorter utterances to pass
            "min_silence_duration": 0.1, # split on shorter silences
            "no_speech_threshold": 0.85,
            "language": DEFAULT_LANGUAGE,
            "use_cuda": torch.cuda.is_available(),
            "num_threads": min(4, os.cpu_count() or 1),
            "simple_diarization": False,  # disable simple alternating speaker diarization
            "speaker_similarity_threshold": 0.75,  # similarity threshold for clustering speakers
            "capture_all_speech": False,  # capture all voices without domain filtering
            "blocked_phrases": [
                "thank you for watching",
                "sous-titrage société radio-canada",
                "takk for att du så på",
                "terima kasih telah menonton",
                "subtitled by",
                "captions",
            ],
        }
        
        # Override defaults with provided config
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Initialize audio processing
        self.sample_rate = self.config["sample_rate"]
        self.chunk_size = self.config["chunk_size"]
        
        # Advanced buffers
        self.main_buffer = AudioBuffer(self.sample_rate, self.config["buffer_size"])
        
        # Processing state
        self.vad_state = {
            "in_speech": False,
            "silence_start": None,
            "speech_start": None
        }
        
        # Queues for multi-threaded processing
        self.audio_queue = queue.Queue(maxsize=100)
        self.vad_queue = queue.Queue(maxsize=100)
        self.diarization_queue = queue.Queue(maxsize=50)
        self.transcription_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue()
        
        # Thread pools
        self.executors = {}
        
        # Speaker tracking
        self.speakers = {}  # Dict of SpeakerProfile objects
        self.current_speaker = None
        self.transcript_history = []
        # Track seen transcripts to suppress repeats in a session
        self.seen_texts = set()
        
        # Session state
        self.is_running = False
        self.start_time = None
        
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all ML models with optimized settings."""
        logger.info("Initializing models...")
        
        # Set compute device
        self.device = "cuda" if torch.cuda.is_available() and self.config["use_cuda"] else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Initialize models sequentially to avoid threading issues
        self.initialize_diarization()
        self.initialize_whisper()
        
        logger.info(f"All models initialized successfully")
    
    def initialize_whisper(self):
        """Initialize optimized Whisper large-v3 model."""
        logger.info("Loading Whisper large-v3 model...")
        
        model_id = "openai/whisper-large-v3"
        
        try:
            # Optimized loading with attention implementation autodetection
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with optimized settings
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Optimize for inference
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            # Create optimized pipeline - avoid using device parameter with accelerate
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                chunk_length_s=10,         # 10s chunk for robust context
                stride_length_s=2,         # 2s overlap for smooth segment transitions
                batch_size=1,
                return_timestamps=True,
                return_language=True
            )
            
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def initialize_diarization(self):
        """Initialize NVIDIA NeMo speaker diarization models."""
        logger.info("Initializing NeMo diarization models...")
        
        try:
            # Load speaker embedding model
            self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name="titanet_large"
            )
            
            # Move to appropriate device and optimize
            if self.device == "cuda":
                self.speaker_model = self.speaker_model.to(self.device)
            self.speaker_model.eval()  # Set to evaluation mode
            
            # Load VAD model
            self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                model_name="vad_multilingual_marblenet"
            )
            
            # Move to appropriate device and optimize
            if self.device == "cuda":
                self.vad_model = self.vad_model.to(self.device)
            self.vad_model.eval()  # Set to evaluation mode
            
            logger.info("NeMo diarization models loaded successfully")
            
            # simple_diarization enabled; skip clustering threshold
            self.speaker_embeddings = []
            self.speaker_ids = []
            self.use_nemo = True
            
        except Exception as e:
            logger.error(f"Failed to load NeMo models: {e}")
            
            # Fallback to PyAnnotate if NeMo fails
            try:
                logger.warning("Falling back to PyAnnotate for diarization")
                HF_TOKEN = os.getenv("HF_TOKEN")
                if not HF_TOKEN:
                    raise ValueError("HuggingFace token not found in environment variables")
                
                self.pyannote_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                )
                
                logger.info("PyAnnotate diarization loaded successfully as fallback")
                self.use_nemo = False
            except Exception as e2:
                logger.error(f"Failed to load PyAnnotate: {e2}")
                raise
    
    def audio_capture_thread(self):
        """High-performance audio capture thread."""
        p = pyaudio.PyAudio()
        
        # Configure audio stream with optimal latency settings
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            start=False,  # Don't start yet
            input_device_index=None  # Default input device
        )
        
        # Start the stream
        stream.start_stream()
        logger.info(f"Audio capture started with chunk size: {self.chunk_size}")
        
        try:
            # Main capture loop
            while self.is_running:
                # Read audio chunk
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                
                # Normalize audio
                if np.max(np.abs(audio_chunk)) > 0:
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                
                # Put in queue for processing
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_chunk, block=False)
        except KeyboardInterrupt:
            logger.info("Audio capture stopped by user")
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Audio capture resources released")
    
    def vad_processing_thread(self):
        """Voice Activity Detection processing thread."""
        logger.info("VAD processing thread started")
        
        speech_segments = []
        continuous_silence = 0
        in_speech = False
        current_segment = []
        
        try:
            while self.is_running:
                # Get audio chunk
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process with NeMo VAD
                has_voice = self.detect_voice_activity(audio_chunk)
                
                # Add to main buffer
                self.main_buffer.add(audio_chunk, has_voice)
                
                # Handle voice state tracking
                chunk_duration = len(audio_chunk) / self.sample_rate
                
                if has_voice:
                    if not in_speech:
                        in_speech = True
                        continuous_silence = 0
                        current_segment = [audio_chunk]
                    else:
                        current_segment.append(audio_chunk)
                else:  # No voice
                    continuous_silence += chunk_duration
                    
                    if in_speech:
                        # Still collect during short silences
                        if continuous_silence < self.config["min_silence_duration"]:
                            current_segment.append(audio_chunk)
                        else:
                            # End of speech segment
                            if current_segment and len(current_segment) > 0:
                                # Calculate segment duration
                                segment_audio = np.concatenate(current_segment)
                                segment_duration = len(segment_audio) / self.sample_rate
                                
                                # Only process segments above minimum duration
                                if segment_duration >= self.config["min_voice_duration"]:
                                    if not self.vad_queue.full():
                                        self.vad_queue.put(segment_audio)
                                
                                # Reset
                                current_segment = []
                                in_speech = False
                
                # Complete long speech segments even if still ongoing
                if in_speech and len(current_segment) > 0:
                    segment_audio = np.concatenate(current_segment)
                    segment_duration = len(segment_audio) / self.sample_rate
                    
                    # Process if segment is getting long (>5 seconds)
                    if segment_duration > 5.0:
                        if not self.vad_queue.full():
                            self.vad_queue.put(segment_audio)
                            # Keep the last second to maintain context
                            last_samples = int(1.0 * self.sample_rate)
                            if len(segment_audio) > last_samples:
                                current_segment = [segment_audio[-last_samples:]]
                            else:
                                current_segment = [segment_audio]
                
                self.audio_queue.task_done()
                
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
    
    def detect_voice_activity(self, audio_chunk):
        """Detect voice activity using NeMo VAD."""
        try:
            if self.use_nemo:
                # Use VAD model directly on audio chunk
                with torch.no_grad():
                    # Convert to tensor
                    audio_tensor = torch.tensor(audio_chunk).unsqueeze(0)
                    
                    # NeMo expects shape [B, T]
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    if self.device == "cuda":
                        audio_tensor = audio_tensor.to(self.device)
                    
                    # Get logits
                    logits = self.vad_model.forward(
                        input_signal=audio_tensor, 
                        input_signal_length=torch.tensor([len(audio_chunk)])
                    )
                    
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Check if speech probability exceeds threshold
                    speech_prob = probs[0, 1].item()  # Index 1 is speech class
                    
                    return speech_prob > self.config["silence_threshold"]
            else:
                # Fallback to simple energy-based VAD
                rms = np.sqrt(np.mean(audio_chunk**2))
                return rms > self.config["silence_threshold"]
                
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Default to treating as speech in case of error
            return True
    
    def diarization_thread(self):
        """Speaker diarization processing thread."""
        logger.info("Diarization thread started")
        
        try:
            while self.is_running:
                # Get voice segment
                try:
                    audio_segment = self.vad_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Identify speaker
                speaker_id, embedding = self.identify_speaker(audio_segment)
                
                # Queue for transcription
                self.diarization_queue.put({
                    "audio": audio_segment,
                    "speaker_id": speaker_id,
                    "embedding": embedding,
                    "timestamp": time.time()
                })
                
                self.vad_queue.task_done()
                
        except Exception as e:
            logger.error(f"Diarization error: {e}")
    
    def identify_speaker(self, audio_segment):
        """Identify speaker using NeMo speaker embeddings."""
        try:
            # Simple alternating diarization
            if self.config.get("simple_diarization", False):
                if not self.current_speaker:
                    self.current_speaker = "speaker_0"
                else:
                    self.current_speaker = "speaker_1" if self.current_speaker == "speaker_0" else "speaker_0"
                return self.current_speaker, None

            if self.use_nemo:
                # Create temporary file for NeMo processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                    # Write normalized audio
                    audio_np = audio_segment.astype(np.float32)
                    if np.max(np.abs(audio_np)) > 0:
                        audio_np = audio_np / np.max(np.abs(audio_np))
                    
                    sf.write(tmp_file.name, audio_np, self.sample_rate)
                    
                    # Extract embedding
                    with torch.no_grad():
                        embedding = self.speaker_model.get_embedding(tmp_file.name)
                        embedding = embedding.cpu().numpy()
                
                # Compare with existing speakers
                if len(self.speaker_embeddings) == 0:
                    # First speaker
                    self.speaker_embeddings.append(embedding)
                    self.speaker_ids.append("speaker_0")
                    return "speaker_0", embedding
                
                # Calculate similarities with known speakers
                similarities = []
                for spk_emb in self.speaker_embeddings:
                    sim = np.dot(embedding, spk_emb.T) / (
                        np.linalg.norm(embedding) * np.linalg.norm(spk_emb))
                    similarities.append(sim)
                
                max_sim = max(similarities)
                if max_sim > self.config["speaker_similarity_threshold"]:
                    # Existing speaker
                    speaker_idx = similarities.index(max_sim)
                    speaker_id = self.speaker_ids[speaker_idx]
                    
                    # Update embedding with exponential moving average
                    self.speaker_embeddings[speaker_idx] = 0.8 * self.speaker_embeddings[speaker_idx] + 0.2 * embedding
                    
                    return speaker_id, embedding
                else:
                    # New speaker
                    new_id = f"speaker_{len(self.speaker_embeddings)}"
                    self.speaker_embeddings.append(embedding)
                    self.speaker_ids.append(new_id)
                    return new_id, embedding
            else:
                # Fallback to PyAnnotate
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                    sf.write(tmp_file.name, audio_segment, self.sample_rate)
                    diarization = self.pyannote_pipeline(tmp_file.name)
                    
                    # Extract most likely speaker
                    speaker_counts = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if speaker not in speaker_counts:
                            speaker_counts[speaker] = 0
                        speaker_counts[speaker] += turn.duration
                    
                    # Find speaker with longest duration
                    if speaker_counts:
                        speaker_id = max(speaker_counts.items(), key=lambda x: x[1])[0]
                    else:
                        speaker_id = "unknown"
                    
                    # No embedding in PyAnnotate fallback
                    return speaker_id, None
                
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            return "unknown", None
    
    def transcription_thread(self):
        """Transcription processing thread."""
        logger.info("Transcription thread started")
        while self.is_running:
            # Get diarized segment
            try:
                segment = self.diarization_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Run Whisper transcription
                transcription = self.transcribe_audio(segment["audio"])

                if not transcription or not transcription.strip():
                    continue

                # Filter out blocked phrases
                if any(
                    phrase.lower() in transcription.lower()
                    for phrase in self.config.get("blocked_phrases", [])
                ):
                    logger.info(f"Skipping blocked phrase in transcript: {transcription}")
                    continue

                # Suppress seen or duplicate transcripts
                if transcription in self.seen_texts or (
                    self.transcript_history and transcription == self.transcript_history[-1]["text"]
                ):
                    continue
                self.seen_texts.add(transcription)

                # Determine speaker role
                speaker_role = self.classify_speaker_role(
                    transcription, segment["speaker_id"]
                )
                if speaker_role is None:
                    continue

                # Build and queue result
                result = {
                    "speaker_id": segment["speaker_id"],
                    "role": speaker_role,
                    "text": transcription,
                    "timestamp": segment["timestamp"]
                }
                if segment["speaker_id"] not in self.speakers:
                    self.speakers[segment["speaker_id"]] = SpeakerProfile(
                        id=segment["speaker_id"],
                        embedding=segment["embedding"],
                        role=speaker_role
                    )
                self.transcript_history.append(result)
                self.output_queue.put(result)
            except Exception as e:
                logger.error(f"Transcription error: {e}")
            finally:
                # Mark the segment as processed exactly once
                self.diarization_queue.task_done()
    
    def transcribe_audio(self, audio_segment):
        """Transcribe audio using Whisper large-v3."""
        try:
            audio_np = audio_segment.astype(np.float32)
            # Normalize
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            # Enforce minimum segment duration
            duration = len(audio_np) / self.sample_rate
            if duration < self.config.get("min_voice_duration", 1.0):
                return None
            # Skip very short arrays
            if len(audio_np) < 800:
                return None
            # VAD energy filter
            rms = np.sqrt(np.mean(audio_np**2))
            if rms < self.config["silence_threshold"]:
                return None

            result = self.whisper_pipe(
                audio_np,
                generate_kwargs={
                    "task": "transcribe",
                    "temperature": 0.0,
                    "compression_ratio_threshold": 1.5,  # more repetition filtering
                    "logprob_threshold": 0.0,            # require non-negative logprob
                    "no_speech_threshold": self.config["no_speech_threshold"],
                    "num_beams": 3,                      # beam search for accuracy
                    "max_new_tokens": 256
                }
            )

            # Ignore transcripts in languages other than the configured one
            if "language" in result:
                detected = result["language"]
                if isinstance(detected, dict):
                    code = detected.get("language", detected.get("lang", None))
                else:
                    code = detected
                if code != self.config["language"]:
                    logger.info(f"Ignoring transcript in language: {code}")
                    return None

            if "text" in result:
                text = result["text"].strip()
                return text
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def classify_speaker_role(self, text, speaker_id):
        """Classify speaker as doctor or patient using ML-based approach."""
        # Check if speaker already has a known role
        if speaker_id in self.speakers and self.speakers[speaker_id].role:
            return self.speakers[speaker_id].role
            
        # Prepare text for analysis
        text_lower = text.lower()
        
        # Doctor indicators (medical terminology, questions, recommendations)
        doctor_indicators = [
            "diagnos", "treatment", "prescri", "recommend", "examin", 
            "your condition", "your symptoms", "medical history", "allergies",
            "any pain", "how are you feeling", "follow up", "test results",
            "what brings you", "your medications", "your health", "i'll recommend",
            # broader medical terms for domain-specific segments
            "medical", "specialties", "category", "general practitioner", "family medicine", "cardiology"
        ]
        
        # Patient indicators (personal symptoms, feelings, questions)
        patient_indicators = [
            "i feel", "i'm feeling", "i've been", "it hurts", "my pain",
            "i have a", "i'm having", "my symptoms", "my condition", 
            "worried about", "i noticed", "i wanted to ask", "i'm concerned",
            "i need", "i was wondering", "i've noticed", "will this"
        ]
        
        # Count matches
        doctor_score = sum(1 for term in doctor_indicators if term in text_lower)
        patient_score = sum(1 for term in patient_indicators if term in text_lower)
        
        # Optionally skip transcripts with no domain-specific keywords
        if not self.config.get("capture_all_speech", False):
            if doctor_score == 0 and patient_score == 0:
                return None
        
        # Check context from history
        if len(self.transcript_history) > 0:
            # In medical conversations, doctor and patient usually alternate
            last_entry = self.transcript_history[-1]
            if last_entry["speaker_id"] != speaker_id:
                # Different speaker, likely alternating roles
                if last_entry["role"] == "doctor":
                    patient_score += 2  # Boost patient score
                else:
                    doctor_score += 2  # Boost doctor score
        
        # Determine role
        if doctor_score > patient_score:
            return "doctor"
        elif patient_score > doctor_score:
            return "patient"
        else:
            # Default to doctor for first speaker if uncertain
            if not self.transcript_history:
                return "doctor"
            else:
                # For subsequent unclear cases, keep different role from last speaker
                last_role = self.transcript_history[-1]["role"]
                return "patient" if last_role == "doctor" else "doctor"
    
    def output_thread(self):
        """Handle output formatting and display."""
        logger.info("Output thread started")
        
        try:
            while self.is_running:
                try:
                    result = self.output_queue.get(timeout=0.5)
                    
                    # Format output
                    role = result["role"].upper()
                    text = result["text"]
                    timestamp = result["timestamp"] - self.start_time
                    min_sec = divmod(int(timestamp), 60)
                    
                    # Print with timestamp
                    print(f"[{min_sec[0]:02d}:{min_sec[1]:02d}] {role}: {text}")
                    
                    self.output_queue.task_done()
                except queue.Empty:
                    continue
                
        except Exception as e:
            logger.error(f"Output error: {e}")
    
    def calibrate_ambient_noise(self):
        """Record ambient noise for calibration_duration and adjust silence_threshold."""
        duration = self.config.get("calibration_duration", 0.0)
        factor = self.config.get("calibration_factor", 1.0)
        if duration <= 0:
            return
        logger.info(f"Calibrating ambient noise for {duration}s...")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        frames = []
        n_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(n_chunks):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(audio_chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()
        ambient_audio = np.concatenate(frames) if frames else np.array([], dtype=np.float32)
        rms = np.sqrt(np.mean(ambient_audio**2)) if ambient_audio.size > 0 else 0.0
        new_thresh = rms * factor
        self.config["silence_threshold"] = float(new_thresh)
        logger.info(f"Ambient RMS={rms:.6f}, setting silence_threshold={new_thresh:.6f}")
    
    def start(self):
        """Start the enhanced real-time transcription system."""
        logger.info("Starting enhanced real-time transcription system")
        
        # Perform ambient noise calibration if configured
        self.calibrate_ambient_noise()

        # Initialize state
        self.is_running = True
        self.start_time = time.time()
        
        # Create thread pools
        self.executors["thread_pool"] = ThreadPoolExecutor(
            max_workers=self.config["num_threads"]
        )
        
        # Start worker threads
        threads = {
            "audio_capture": threading.Thread(
                target=self.audio_capture_thread, 
                daemon=True
            ),
            "vad_processing": threading.Thread(
                target=self.vad_processing_thread,
                daemon=True
            ),
            "diarization": threading.Thread(
                target=self.diarization_thread,
                daemon=True
            ),
            "transcription": threading.Thread(
                target=self.transcription_thread,
                daemon=True
            ),
            "output": threading.Thread(
                target=self.output_thread,
                daemon=True
            )
        }
        
        # Start all threads
        for name, thread in threads.items():
            thread.start()
            logger.info(f"Started {name} thread")
        
        print("\n====== Real-Time Medical Transcription ======")
        print("Press Ctrl+C to stop\n")
        
        # Main loop - keep program alive and handle graceful shutdown
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.stop()
        except Exception as e:
            logger.error(f"Main thread error: {e}")
            self.stop()
    
    def stop(self):
        """Stop the transcription system gracefully."""
        self.is_running = False
        
        # Clean up resources
        for executor_name, executor in self.executors.items():
            executor.shutdown(wait=False)
        
        logger.info("Transcription system stopped")
        
        # Print summary
        print("\n====== Session Summary ======")
        print(f"Duration: {time.time() - self.start_time:.1f} seconds")
        print(f"Speakers detected: {len(self.speakers)}")
        print("===========================\n")

if __name__ == "__main__":
    # Configuration (best-practice thresholds)
    config = {
        "language": DEFAULT_LANGUAGE,
        "chunk_size": 4000,
        "sample_rate": 16000,
        "use_cuda": torch.cuda.is_available(),
        "num_threads": min(4, os.cpu_count() or 1),
        "silence_threshold": 0.005,
        "min_voice_duration": 0.2,
        "min_silence_duration": 0.1,
        "no_speech_threshold": 0.85,
        "simple_diarization": False,
        "speaker_similarity_threshold": 0.75,
        "capture_all_speech": True,  # capture all voices in main run
        "blocked_phrases": [
            "thank you for watching",
            "sous-titrage société radio-canada",
            "takk for att du så på",
            "terima kasih telah menonton",
            "subtitled by",
            "captions",
        ],
    }
    
    # Create and start the transcriber
    transcriber = RealTimeTranscriber(config)
    transcriber.start()