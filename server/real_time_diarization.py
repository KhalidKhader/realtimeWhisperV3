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
import re
from langdetect import detect_langs
import math
import argparse

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
    def __init__(self, id, embedding=None):
        self.id = id
        self.embedding = embedding
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

class TranscriptionValidator:
    """
    Simplified validator that just forwards the transcription without complex validation.
    """
    def __init__(self, config=None):
        self.config = config or {}
    
    def validate(self, text, confidence_scores=None):
        """Basic validation just checks if text is not empty."""
        if not text or len(text.strip()) < 2:
            return False, 0.0, "Empty text"
        return True, 1.0, "Valid speech"
        
    def _clean_text(self, text):
        """Basic text cleaning."""
        return text.strip()

class AudioPreprocessor:
    """
    Enhanced audio preprocessing to improve signal quality before transcription.
    """
    def __init__(self, sample_rate=16000, config=None):
        self.sample_rate = sample_rate
        self.config = {
            "noise_reduction_factor": 0.5,   # Noise reduction strength
            "normalize_audio": True,         # Apply normalization
            "apply_filters": True,           # Apply audio filtering
            "high_pass_cutoff": 100,         # High-pass filter cutoff (Hz)
            "low_pass_cutoff": 3500,         # Low-pass filter cutoff (Hz)
            "energy_threshold": 0.005,       # Energy threshold for voice
            "dynamic_range_compression": 0.7 # Dynamic range compression factor
        }
        
        if config:
            self.config.update(config)
            
        # Pre-calculate filter coefficients
        self._init_filters()
        
        # Keep noise profile for adaptive filtering
        self.noise_profile = None
        self.noise_samples = []
        self.noise_std = 0.01  # Initial estimate
    
    def _init_filters(self):
        """Initialize filter coefficients for efficient processing."""
        # This is a simplified implementation - in a full implementation we'd use scipy.signal
        # to create proper butterworth filters for high-pass and low-pass
        pass
    
    def update_noise_profile(self, audio_chunk):
        """Update the noise profile based on low-energy audio chunks."""
        energy = np.mean(audio_chunk**2)
        if energy < self.config["energy_threshold"]:
            # Likely noise, update noise profile
            self.noise_samples.append(audio_chunk)
            if len(self.noise_samples) > 10:
                self.noise_samples.pop(0)  # Keep only recent noise samples
                
            # Update noise standard deviation
            if len(self.noise_samples) > 3:
                concat_noise = np.concatenate(self.noise_samples)
                self.noise_std = np.std(concat_noise)
    
    def process(self, audio_chunk):
        """Apply comprehensive preprocessing to audio chunk."""
        if not self.config["apply_filters"]:
            return audio_chunk
            
        # Update noise profile
        self.update_noise_profile(audio_chunk)
        
        # Apply sequence of enhancements
        audio = audio_chunk.copy()
        
        # 1. Normalize if needed
        if self.config["normalize_audio"] and np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        # 2. Apply spectral subtraction for noise reduction
        if self.noise_std > 0 and self.config["noise_reduction_factor"] > 0:
            noise_scale = self.noise_std * self.config["noise_reduction_factor"]
            # Simple noise gate
            audio[np.abs(audio) < noise_scale * 2] = 0
            
        # 3. Apply dynamic range compression
        if self.config["dynamic_range_compression"] < 1.0:
            # Simple compression
            compressed = np.sign(audio) * np.abs(audio)**self.config["dynamic_range_compression"]
            # Normalize to maintain relative volume
            if np.max(np.abs(compressed)) > 0:
                compressed = compressed / np.max(np.abs(compressed))
            audio = compressed
            
        return audio

class RealTimeTranscriber:
    def __init__(self, config=None):
        # Default configuration thresholds tuned for clarity
        default_config = {
            "sample_rate": 16000,
            "chunk_size": 6000,        # Larger chunks (625ms) for better context
            "buffer_size": 5,         # 30s context buffer
            "silence_threshold": 0.01, # Higher threshold for cleaner speech detection
            "min_voice_duration": 0.7,   # Longer segments for better quality
            "min_silence_duration": 0.3, # Longer silence for better segmentation
            "language": DEFAULT_LANGUAGE,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "num_threads": min(8, os.cpu_count() or 2),  # use more threads if available
            "max_speakers": 3,  # Fewer speakers for clearer diarization
            "speaker_similarity_threshold": 0.85,  # Higher threshold for better speaker distinction
            "capture_all_speech": True  # capture all voices in main run
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
        self.initialization_complete = False
        
        # Speaker tracking system
        self.speaker_labels = {}  # Maps cluster IDs to speaker IDs
        self.next_speaker_id = 0  # Counter for generating speaker IDs
        
        # Initialize validation components - much simpler now
        self.validator = TranscriptionValidator(config=self.config)
        
        # Initialize models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all ML models with optimized settings."""
        logger.info("Initializing models...")
        print("Initializing models...")
        
        # Set compute device - prioritize MPS
        if self.config["use_mps"] and torch.backends.mps.is_available():
            self.device = "mps"
        elif self.config["use_cuda"] and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        print(f"Using device: {self.device} with dtype: {self.torch_dtype}")
        
        # Initialize models sequentially to avoid threading issues
        print("Initializing diarization...")
        self.initialize_diarization()
        print("Diarization initialized. Initializing Whisper...")
        self.initialize_whisper()
        print("Whisper initialized.")
        
        # Mark model initialization as complete
        self.initialization_complete = True
        logger.info(f"All models initialized successfully on {self.device}")
        print(f"All models initialized successfully on {self.device}")
    
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
            
            # Optimize for inference - move to the selected device
            self.model = self.model.to(self.device)
            logger.info(f"Model moved to {self.device} device")
            
            # Create optimized pipeline
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                chunk_length_s=10,         # 10s chunk for robust context
                stride_length_s=3,         # 3s overlap for better continuity
                batch_size=1
            )
            
            # Fixed generate parameters that are compatible with Whisper v3
            self.generate_kwargs = {
                "task": "transcribe",
                "language": self.config.get("language", "en"),
                "temperature": 0.0,
                "compression_ratio_threshold": 1.5,
                "logprob_threshold": -0.7,
                "no_speech_threshold": 0.6
            }
            
            logger.info(f"Whisper model loaded successfully on {self.device}")
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
            
            # Load VAD model
            self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
                model_name="vad_multilingual_marblenet"
            )
            
            # For MPS compatibility, move speaker model to MPS but keep VAD on CPU
            # This ensures maximum compatibility while still using GPU acceleration where possible
            if self.device == "mps":
                try:
                    # Move speaker model to MPS for faster embedding extraction
                    self.speaker_model = self.speaker_model.to("mps")
                    # Always keep VAD on CPU for compatibility
                    self.vad_model = self.vad_model.to("cpu")
                    logger.info("Speaker model on MPS, VAD model on CPU for MPS compatibility")
                except Exception as e:
                    logger.warning(f"Failed to move models to MPS: {e}")
                    # Fall back to CPU for everything
                    self.speaker_model = self.speaker_model.to("cpu")
                    self.vad_model = self.vad_model.to("cpu")
                    logger.info("All NeMo models on CPU due to MPS compatibility issues")
            elif self.device == "cuda":
                # On CUDA, we can use GPU for everything
                self.speaker_model = self.speaker_model.to("cuda")
                self.vad_model = self.vad_model.to("cuda")
                logger.info("All NeMo models on CUDA")
            else:
                # Default to CPU
                self.speaker_model = self.speaker_model.to("cpu")
                self.vad_model = self.vad_model.to("cpu")
                logger.info("All NeMo models on CPU")
            
            # Set to evaluation mode
            self.speaker_model.eval()
            self.vad_model.eval()
            
            # Log the actual devices being used
            logger.info(f"Speaker model device: {next(self.speaker_model.parameters()).device}")
            logger.info(f"VAD model device: {next(self.vad_model.parameters()).device}")
            
            # Initialize clustering parameters for diarization
            try:
                # Import NeMo clustering and diarization components
                from nemo.collections.asr.parts.utils.offline_clustering import (
                    SpeakerClustering
                )

                # Get clustering parameters from config
                clustering_config = self.config.get("clustering", {})
                min_samples = clustering_config.get("min_samples", 2)
                eps = clustering_config.get("eps", 0.15)
                max_speakers = clustering_config.get("max_speakers", 8)
                enhanced = clustering_config.get("enhanced", True)
                
                # Initialize clustering object for dynamic usage
                self.clustering = SpeakerClustering(
                    min_samples=min_samples,
                    eps=eps, 
                    oracle_num_speakers=False,
                    max_num_speakers=max_speakers, 
                    enhanced=enhanced,
                    metric='cosine'
                )
                
                self.use_nemo_clustering = True
                logger.info("NeMo clustering initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load NeMo clustering modules: {e}")
                self.use_nemo_clustering = False
            
            # Speaker tracking state
            self.speaker_embeddings_buffer = []  # Store (embedding, timestamp) tuples
            self.speaker_labels = {}  # Mapping from NeMo cluster IDs to speaker_ids
            self.next_speaker_id = 0
            
            logger.info("NeMo diarization models loaded successfully")
            
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
        print("VAD processing thread started - ready to detect speech")
        
        speech_segments = []
        continuous_silence = 0
        in_speech = False
        current_segment = []
        
        # Enhanced parameters for better voice capture
        max_segment_length = 10.0  # maximum segment length in seconds before forced processing
        min_segment_energy = 0.001  # minimum energy to consider for processing
        last_detection_time = time.time()
        detection_counter = 0
        
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
                
                # Diagnostic output periodically
                if has_voice:
                    now = time.time()
                    detection_counter += 1
                    if now - last_detection_time > 5.0:
                        print(f"Speech detected ({detection_counter} chunks since last report)")
                        last_detection_time = now
                        detection_counter = 0
                
                # Handle voice state tracking
                chunk_duration = len(audio_chunk) / self.sample_rate
                
                if has_voice:
                    if not in_speech:
                        in_speech = True
                        continuous_silence = 0
                        current_segment = [audio_chunk]
                        print("Speech segment started")
                    else:
                        current_segment.append(audio_chunk)
                else:  # No voice
                    continuous_silence += chunk_duration
                    
                    if in_speech:
                        # Still collect during short silences
                        if continuous_silence < self.config.get("min_silence_duration", 0.05):
                            current_segment.append(audio_chunk)
                        else:
                            # End of speech segment
                            if current_segment and len(current_segment) > 0:
                                # Calculate segment duration
                                segment_audio = np.concatenate(current_segment)
                                segment_duration = len(segment_audio) / self.sample_rate
                                
                                # Only process segments above minimum duration
                                if segment_duration >= self.config.get("min_voice_duration", 0.1):
                                    print(f"Speech segment (duration: {segment_duration:.2f}s) sent for processing")
                                    if not self.vad_queue.full():
                                        self.vad_queue.put(segment_audio)
                                    else:
                                        print("WARNING: VAD queue full, dropping segment")
                                else:
                                    print(f"Speech segment too short ({segment_duration:.2f}s), dropping")
                                
                                # Reset
                                current_segment = []
                                in_speech = False
                
                # Complete long speech segments even if still ongoing
                if in_speech and len(current_segment) > 0:
                    segment_audio = np.concatenate(current_segment)
                    if segment_audio.size == 0:
                        continue  # Skip empty segments 
                    segment_duration = len(segment_audio) / self.sample_rate
                    
                    # Process if segment is getting long (>5 seconds)
                    if segment_duration > max_segment_length:
                        # Check if segment has minimum energy to be worth processing
                        rms = np.sqrt(np.mean(segment_audio**2)) if segment_audio.size > 0 else 0
                        if rms < min_segment_energy:
                            print(f"Skipping low-energy segment: {rms:.6f}")
                            current_segment = []
                            continue
                        
                        print(f"Long speech segment (duration: {segment_duration:.2f}s) sent for processing")
                        if not self.vad_queue.full():
                            try:
                                self.vad_queue.put(segment_audio, block=False)
                            except queue.Full:
                                print("WARNING: VAD queue full, dropping segment")
                                continue
                            # Keep the last second to maintain context
                            last_samples = int(2.0 * self.sample_rate)  # 2 seconds of context overlap
                            if len(segment_audio) > last_samples:
                                current_segment = [segment_audio[-last_samples:]]
                            else:
                                current_segment = [segment_audio]
                
                self.audio_queue.task_done()
                
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            print(f"VAD processing error: {e}")
    
    def detect_voice_activity(self, audio_chunk, threshold=0.0002):
        """
        Improved Voice Activity Detection (VAD) using energy-based and spectral features.
        
        Args:
            audio_chunk: Audio numpy array
            threshold: Energy threshold for voice detection
            
        Returns:
            bool: True if voice detected, False otherwise
        """
        try:
            # Convert to numpy array if necessary
            if not isinstance(audio_chunk, np.ndarray):
                audio_chunk = np.array(audio_chunk)
                
            # Ensure audio is in correct shape and type
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.mean(axis=1)  # Convert stereo to mono if needed
            
            # 1. Energy-based detection
            energy = np.mean(np.square(audio_chunk))
            energy_vad = energy > threshold
            
            if not energy_vad:
                return False
                
            # 2. Spectral features for improved detection
            if len(audio_chunk) >= 512:  # Minimum size for FFT
                # Calculate spectral centroid and flux
                fft = np.abs(np.fft.rfft(audio_chunk))
                freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
                
                # Ignore very low frequencies (below 100Hz) which are often noise
                mask = freqs > 50  # Lower this from 100 to 50 Hz to be more sensitive
                if np.sum(mask) > 0:
                    fft_filtered = fft[mask]
                    
                    # Check if there's significant mid-range frequency content (speech is ~300-3000Hz)
                    speech_range_mask = (freqs > 200) & (freqs < 4000)  # Widen speech range
                    if np.sum(speech_range_mask) > 0:
                        speech_energy = np.mean(fft[speech_range_mask])
                        background_energy = np.mean(fft[~speech_range_mask])
                        
                        # If speech energy is significantly higher than background
                        spectral_speech_detected = speech_energy > (1.2 * background_energy)  # Lower this from 1.5 to 1.2
                        return spectral_speech_detected
            
            # Default to energy-based decision if spectral analysis is not conclusive
            return energy_vad
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            # Default to basic threshold in case of error
            return np.mean(np.abs(audio_chunk)) > threshold
    
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
        """Identify speaker using NeMo speaker embeddings and similarity matching."""
        try:
            # Get max speakers and similarity threshold
            max_speakers = self.config.get("max_speakers", 3)
            similarity_threshold = self.config.get("speaker_similarity_threshold", 0.7)
            
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
                        try:
                            embedding = self.speaker_model.get_embedding(tmp_file.name)
                            # Move to CPU and convert to numpy for safe processing
                            embedding = embedding.cpu().numpy()
                            
                            # Ensure embedding is properly flattened (1D array)
                            if embedding.ndim > 2:
                                # If we have a 3D tensor, we need to flatten it to 2D for similarity computation
                                logger.warning(f"Got embedding with shape {embedding.shape}, flattening")
                                embedding = embedding.reshape(1, -1).squeeze()
                            elif embedding.ndim < 2:
                                # Ensure 2D for similarity computation
                                embedding = embedding.reshape(1, -1)
                        except Exception as e:
                            logger.error(f"Error extracting embedding: {e}")
                            # Create a fallback embedding
                            embedding = np.random.rand(512).astype(np.float32)  # Standard 512-dim embedding size
                
                # Ensure embeddings are consistently shaped
                if embedding.ndim == 1:
                    embedding_shaped = embedding.reshape(1, -1)
                else:
                    embedding_shaped = embedding
                
                # If no speakers yet, create first speaker
                if len(self.speakers) == 0:
                    new_id = "Speaker_1"
                    self.speakers[new_id] = SpeakerProfile(id=new_id, embedding=embedding_shaped)
                    return new_id, embedding_shaped
                    
                # If we have existing speakers, calculate similarities
                if self.speakers:
                    similarities = []
                    for speaker_id, profile in self.speakers.items():
                        if profile.embedding is not None:
                            try:
                                # Ensure both embeddings are 2D arrays
                                emb1 = embedding_shaped
                                emb2 = profile.embedding
                                
                                if emb2.ndim == 1:
                                    emb2 = emb2.reshape(1, -1)
                                
                                # Compute similarity
                                sim = cosine_similarity(emb1, emb2)[0][0]
                                similarities.append((speaker_id, sim))
                                print(f"Similarity with {speaker_id}: {sim:.3f}")
                            except Exception as e:
                                logger.error(f"Error computing similarity: {e}")
                                similarities.append((speaker_id, 0.0))
                    
                    # Sort by similarity (highest first)
                    if similarities:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        best_match_id, best_sim = similarities[0]
                        
                        # If found a good match, return that speaker
                        if best_sim > similarity_threshold:
                            # Update speaker's embedding with a weighted average
                            self.speakers[best_match_id].update_embedding(embedding_shaped, 0.3)
                            print(f"Speaker {best_match_id} matched with similarity {best_sim:.3f}")
                            return best_match_id, embedding_shaped
                        
                        # If we've reached max speakers, return the most similar one anyway
                        if len(self.speakers) >= max_speakers:
                            # Update speaker's embedding slightly
                            self.speakers[best_match_id].update_embedding(embedding_shaped, 0.1)
                            print(f"Max speakers reached. Using {best_match_id} (similarity: {best_sim:.3f})")
                            return best_match_id, embedding_shaped
                
                # Create a new speaker profile if below the max limit
                if len(self.speakers) < max_speakers:
                    new_id = f"Speaker_{len(self.speakers) + 1}"
                    self.speakers[new_id] = SpeakerProfile(id=new_id, embedding=embedding_shaped)
                    print(f"New speaker detected: {new_id}")
                    return new_id, embedding_shaped
                
                # Fallback to the most similar speaker
                if similarities:
                    best_match_id = similarities[0][0]
                    return best_match_id, embedding_shaped
                
                # Ultimate fallback
                return f"Speaker_1", None
                
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            # Return a default speaker ID
            return "Speaker_1", None
    
    def transcription_thread(self):
        """Transcription processing thread."""
        logger.info("Transcription thread started")
        print("Transcription thread started - ready to transcribe speech")
        segment_counter = 0
        
        while self.is_running:
            # Get diarized segment
            try:
                segment = self.diarization_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                segment_counter += 1
                print(f"Processing segment #{segment_counter} from speaker {segment['speaker_id']}")
                
                # Run Whisper transcription
                start_time = time.time()
                transcription = self.transcribe_audio(segment["audio"])
                trans_time = time.time() - start_time
                
                print(f"Transcription completed in {trans_time:.2f}s")

                # Skip if no valid transcription was found
                if not transcription or not transcription.strip():
                    print("No transcription result - empty or None")
                    continue

                # Exact match duplicate detection - only check exact string matches
                # to avoid over-filtering similar content
                if transcription in self.seen_texts or (
                    self.transcript_history and transcription == self.transcript_history[-1]["text"]
                ):
                    print(f"Duplicate transcription - skipping: '{transcription}'")
                    continue
                    
                # Add to seen texts only if we'll use this transcription
                self.seen_texts.add(transcription)

                # Use speaker ID directly without role classification
                speaker_id = segment["speaker_id"]

                # Build and queue result
                result = {
                    "speaker_id": speaker_id,
                    "role": speaker_id,  # Use speaker_id as the role
                    "text": transcription,
                    "timestamp": segment["timestamp"],
                    "confidence": 1.0  # Store confidence score for later analysis
                }
                
                if speaker_id not in self.speakers:
                    self.speakers[speaker_id] = SpeakerProfile(
                        id=speaker_id,
                        embedding=segment["embedding"]
                    )
                
                self.transcript_history.append(result)
                self.output_queue.put(result)
                print(f"Transcript added to output queue: {speaker_id}: {transcription}")
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                print(f"Transcription error: {e}")
            finally:
                # Mark the segment as processed exactly once
                self.diarization_queue.task_done()
    
    def transcribe_audio(self, audio_segment):
        """Transcribe audio using Whisper large-v3."""
        try:
            # Process audio through Whisper pipeline
            audio_np = audio_segment.astype(np.float32)
            
            # Skip processing if segment is too short
            if len(audio_np) < 1600:  # At least 100ms of audio
                return None

            # Normalize audio (critical for good results)
            if np.max(np.abs(audio_np)) > 0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            
            # Apply some basic noise reduction if energy is low
            noise_floor = 0.005
            audio_np[np.abs(audio_np) < noise_floor] = 0
                
            # Run transcription
            result = self.whisper_pipe(
                audio_np,
                generate_kwargs=self.generate_kwargs,
                return_timestamps=False
            )

            # Extract text
            if "text" in result:
                text = result["text"].strip()
                if text:
                    return text
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def output_thread(self):
        """Handle output formatting and display."""
        logger.info("Output thread started")
        print("Output thread started - ready to display transcriptions")
        output_counter = 0
        
        try:
            while self.is_running:
                try:
                    result = self.output_queue.get(timeout=0.5)
                    output_counter += 1
                    
                    # Format output with speaker ID and confidence score
                    speaker_label = result["speaker_id"]
                    text = result["text"]
                    timestamp = result["timestamp"] - self.start_time
                    min_sec = divmod(int(timestamp), 60)
                    
                    # Add confidence score to output if available
                    confidence_str = ""
                    if "confidence" in result and result["confidence"] is not None:
                        confidence = result["confidence"]
                        confidence_str = f"[{confidence:.2f}]"
                    
                    # Print with timestamp and confidence
                    output_line = f"[{min_sec[0]:02d}:{min_sec[1]:02d}] {speaker_label}{confidence_str}: {text}"
                    print(f"OUTPUT #{output_counter}: {output_line}")
                    
                    self.output_queue.task_done()
                except queue.Empty:
                    continue
                
        except Exception as e:
            logger.error(f"Output error: {e}")
            print(f"Output error: {e}")
    
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
        
        try:
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
        except Exception as e:
            logger.error(f"Error during start: {e}")
            self.stop()
    
    def stop(self):
        """Stop the transcription system gracefully."""
        # Only stop if running (avoid multiple stops)
        if self.is_running:
            self.is_running = False
            
            # Clean up resources
            for executor_name, executor in self.executors.items():
                try:
                    executor.shutdown(wait=False)
                except Exception as e:
                    logger.error(f"Error shutting down executor {executor_name}: {e}")
            
            logger.info("Transcription system stopped")
            
            # Print summary
            print("\n====== Session Summary ======")
            total_time = time.time() - self.start_time if self.start_time else 0
            print(f"Duration: {total_time:.1f} seconds")
            print(f"Speakers detected: {len(self.speakers)}")
            print("===========================\n")
        else:
            logger.info("Transcription system already stopped")

    def is_complete_sentence(self, text):
        """
        Determine if text is likely a complete sentence based on linguistic patterns.
        Returns True if the text appears to be a complete sentence, False otherwise.
        """
        # Strip whitespace and check if empty
        text = text.strip()
        if not text:
            return False
            
        # Check if text ends with sentence-ending punctuation
        sentence_endings = [".", "!", "?", "...", ":", ";"]
        has_ending_punct = any(text.endswith(end) for end in sentence_endings)
        
        # Check for complete sentence structure by looking for verb phrases
        # Basic verb patterns (not exhaustive but catches common cases)
        common_verbs = ["is", "are", "was", "were", "have", "has", "had", "do", "does", "did",
                        "can", "could", "will", "would", "should", "may", "might", "must",
                        "take", "make", "go", "see", "know", "get", "feel", "think", "come",
                        "look", "want", "give", "use", "find", "need", "try", "ask", "tell",
                        "work", "seem", "call", "continue", "visit", "prescribe", "recommend"]
                        
        # Check for presence of a verb (simplistic but effective first pass)
        words = text.lower().split()
        has_verb = any(verb in words for verb in common_verbs)
        
        # Check for imperative sentences (commands) which often start with verbs
        starts_with_verb = words[0] in common_verbs if words else False
        
        # Special cases for conversational fragments that are acceptable
        ok_fragments = [
            "Yes", "No", "Maybe", "Sure", "Thanks", "Thank you", "OK", "Okay",
            "I see", "Of course", "Absolutely", "Definitely", "Certainly",
            "Not really", "I agree", "I understand", "Go ahead", "I'm sorry"
        ]
        
        is_ok_fragment = any(text.lower().startswith(frag.lower()) for frag in ok_fragments)
        
        # Detect likely sentence fragments (incomplete thoughts)
        fragment_starts = ["and then", "so that", "which is", "because", "even though", "although"]
        is_dependent_clause = any(text.lower().startswith(start) for start in fragment_starts)
        
        # We consider a sentence complete if one of these is true:
        # 1. It ends with proper punctuation AND either has a verb or is an imperative
        # 2. It's a recognized conversational fragment
        # 3. It's longer than 5 words and has a verb (even without punctuation)
        
        return (
            (has_ending_punct and (has_verb or starts_with_verb)) or
            is_ok_fragment or
            (len(words) > 5 and has_verb and not is_dependent_clause)
        )
        
    def filter_incomplete_sentences(self, text, is_final=False):
        """
        Filter out incomplete sentences from transcription text.
        If is_final is True, returns the text regardless as it's the final version.
        """
        if is_final:
            return text
            
        if not text:
            return ""
            
        # Split into potential sentences
        # This uses a simple split on punctuation, which is not perfect but works for most cases
        sentence_endings = [". ", "! ", "? ", ".\n", "!\n", "?\n", "... "]
        for ending in sentence_endings:
            text = text.replace(ending, ending + "<SPLIT>")
            
        sentences = text.split("<SPLIT>")
        
        # If only one sentence, check if it's complete
        if len(sentences) == 1:
            return text if self.is_complete_sentence(text) else ""
            
        # For multiple sentences, keep all complete sentences 
        # plus the last one if we're at the end of a segment
        result_sentences = []
        for i, sentence in enumerate(sentences):
            if i == len(sentences) - 1:  # Last sentence
                if self.is_complete_sentence(sentence) or is_final:
                    result_sentences.append(sentence)
            else:
                if sentence.strip():  # Non-empty
                    result_sentences.append(sentence)
                    
        return "".join(result_sentences)

    def remove_repeated_phrases(self, text):
        """
        Detect and remove repeated phrases in transcribed text.
        """
        if not text or len(text) < 10:
            return text
            
        # First pass: Remove exact duplicated sentences back-to-back
        # Split the text into sentences
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in '.!?' and (len(current) > 1):
                sentences.append(current.strip())
                current = ""
        if current:
            sentences.append(current.strip())
            
        # Remove consecutive duplicate sentences
        i = 0
        result_sentences = []
        while i < len(sentences):
            result_sentences.append(sentences[i])
            # Skip ahead past duplicates
            j = i + 1
            while j < len(sentences) and sentences[j] == sentences[i]:
                j += 1
            i = j
            
        # Second pass: Remove repeated phrases within the text (like stuttering)
        result_text = ' '.join(result_sentences)
        
        # Find phrases that repeat more than twice
        words = result_text.split()
        if len(words) < 6:  # Skip short texts
            return result_text
            
        # Look for repeated word patterns (2-4 words long)
        for pattern_len in range(2, min(5, len(words) // 2)):
            i = 0
            while i <= len(words) - pattern_len * 2:
                pattern1 = ' '.join(words[i:i+pattern_len])
                pattern2 = ' '.join(words[i+pattern_len:i+pattern_len*2])
                
                # If we found a repeated pattern
                if pattern1.lower() == pattern2.lower():
                    # Remove the second occurrence
                    words = words[:i+pattern_len] + words[i+pattern_len*2:]
                else:
                    i += 1
        
        return ' '.join(words)
        
    def clean_repetitive_answers(self, text):
        """
        Clean up repetitive short answers like "Yes. Yes. Yes."
        """
        common_answers = ["yes", "no", "maybe", "okay", "sure", "right", "exactly", 
                         "correct", "thanks", "thank you", "ok", "all right", "uh-huh",
                         "mm-hmm", "got it", "i see", "understood"]
        
        # Check each common answer for repetition
        for answer in common_answers:
            # Look for variations with different endings
            for ending in ["", ".", "!", "?"]:
                pattern = f"{answer}{ending}"
                if pattern.lower() in text.lower():
                    # Count occurrences (case insensitive)
                    count = 0
                    lower_text = text.lower()
                    lower_pattern = pattern.lower()
                    
                    # Count non-overlapping occurrences
                    pos = 0
                    while True:
                        pos = lower_text.find(lower_pattern, pos)
                        if pos == -1:
                            break
                        count += 1
                        pos += len(lower_pattern)
                    
                    # If repeated more than once, replace with just one instance
                    if count > 1:
                        # Find the first occurrence with proper casing
                        pattern_pos = lower_text.find(lower_pattern)
                        original_casing = text[pattern_pos:pattern_pos+len(pattern)]
                        
                        # Remove all instances and add back just one
                        text = re.sub(f"(?i){re.escape(pattern)}\\s*", "", text)
                        text = original_casing + " " + text.strip()
        
        return text.strip()

    def postprocess_text(self, text):
        """Apply various post-processing rules to clean up the transcribed text."""
        if not text or len(text) < 5:
            return text
        
        # De-duplicate repetitions at different levels
        text = self.remove_repeated_phrases(text)
        text = self.clean_repetitive_answers(text)
        
        # Fix common medical transcription errors
        medical_terms = {
            r'\bblood pressure\b': 'blood pressure',
            r'\bheart rate\b': 'heart rate', 
            r'\bpulse\b': 'pulse',
            r'\brespiration\b': 'respiration',
            r'\bsystolic\b': 'systolic',
            r'\bdiastolic\b': 'diastolic',
            r'\btemperature\b': 'temperature',
            r'\boxygen saturation\b': 'oxygen saturation',
            r'\bO2 sat\b': 'O2 sat',
            r'\bcholesterol\b': 'cholesterol',
            r'\bglucose\b': 'glucose',
            r'\bA1C\b': 'A1C',
            r'\bmedication\b': 'medication',
            r'\bprescription\b': 'prescription',
            r'\bdiagnosis\b': 'diagnosis',
            r'\bsymptoms?\b': 'symptom',
            r'\bpatient\b': 'patient',
            r'\bdoctor\b': 'doctor',
        }
        
        # Capitalize standard medical terms for improved readability
        for term, replacement in medical_terms.items():
            pattern = re.compile(term, re.IGNORECASE)
            text = pattern.sub(replacement, text)
        
        # Fix spacing after punctuation
        text = re.sub(r'(\w)([,.!?;:])', r'\1\2 ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1 ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Ensure first character is capitalized (sentence starts)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        # Ensure proper spacing around sentences
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix common transcription artifacts
        filler_words = [
            r'\bum+\b', r'\buh+\b', r'\ber+\b', r'\bah+\b', 
            r'\blike\b(?!\s+to|\s+the|\s+a)', r'\bso\b\s+\bso\b', 
            r'\byou\s+know\b(?!\s+what|\s+that|\s+how|\s+if|\s+when)'
        ]
        
        for word in filler_words:
            text = re.sub(word, '', text, flags=re.IGNORECASE)
        
        # Clean up post-removal spacing issues
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix spacing after commas, periods, etc.
        text = re.sub(r'(\w)([,.!?;:])(\w)', r'\1\2 \3', text)
        
        return text

    def _get_best_device(self):
        """Select the best available compute device."""
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) device available, using it")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("CUDA device available, using it")
            return "cuda"
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"

    def process_chunk(self, chunk, chunk_duration):
        """Process a single audio chunk through the full pipeline."""
        # This is a convenience method for direct API access or testing
        # It returns the processed result directly instead of queueing
        
        # Preprocess audio
        processed_audio = self.audio_preprocessor.process(chunk)
        
        # Detect voice activity
        has_voice = self.detect_voice_activity(processed_audio)
        if not has_voice:
            return None
            
        # Transcribe
        transcription = self.transcribe_audio(processed_audio)
        if not transcription:
            return None
            
        # Validate
        is_valid, confidence, reason = self.validator.validate(transcription, None)
        if not is_valid:
            logger.debug(f"Rejected transcription: {reason} - '{transcription}'")
            return None
            
        # Return result for direct processing
        return {
            "text": transcription,
            "confidence": confidence,
            "timestamp": time.time(),
            "duration": chunk_duration
        }

    def validate_transcription(self, text, confidence):
        """
        Simple validation that only filters empty text
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0, "Empty text"
            
        # Use a reasonable confidence threshold
        if confidence is not None and confidence < 0.3:
            return False, confidence, "Very low confidence"
            
        return True, confidence or 0.9, "Passed validation"

if __name__ == "__main__":
    # Add simple argument parsing for key parameters
    parser = argparse.ArgumentParser(description="Real-time audio transcription with speaker diarization")
    parser.add_argument("--language", "-l", default="en", help="Language code (e.g., 'en', 'es', 'fr', etc.)")
    parser.add_argument("--speakers", "-s", type=int, default=2, help="Maximum number of speakers to detect")
    
    args = parser.parse_args()
    
    # Add diagnostic print statement
    print("Starting real-time transcription system...")
    print(f"Language: {args.language}, Number of speakers: {args.speakers}")
    
    # Configuration focused on quality
    config = {
        "language": args.language,          # Use specified language
        "max_speakers": args.speakers,      # Use specified number of speakers
        "use_mps": True,                    # Use Apple Silicon MPS acceleration
        "silence_threshold": 0.01,          # Balanced voice detection threshold
        "chunk_size": 6000,                 # Larger chunks for better context (500ms)
        "min_voice_duration": 0.7,          # Minimum speech segment for quality
        "min_silence_duration": 0.3,        # Better segmentation between utterances
        "speaker_similarity_threshold": 0.9, # Higher threshold for better speaker distinction
    }
    
    print("Creating transcriber object...")
    # Create and start the transcriber
    try:
        transcriber = RealTimeTranscriber(config)
        print("Transcriber created successfully. Starting...")
        transcriber.start()
    except Exception as e:
        print(f"ERROR: Failed to start transcriber: {e}")
        import traceback
        traceback.print_exc()

