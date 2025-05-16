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
import time
import logging
import queue
import threading
import numpy as np
import torch
import torch.nn.functional as F
import pyaudio
import soundfile as sf
import wave
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline as PyannotePipeline
import tempfile
from concurrent.futures import ThreadPoolExecutor
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

class SileroVAD:
    """
    Integrates Silero VAD for high-accuracy voice activity detection.
    Uses pre-trained Silero models from PyTorch Hub to filter audio segments.
    """
    
    def __init__(self, config=None):
        """Initialize the Silero VAD model and configurations."""
        self.config = {
            "threshold": 0.35,           # Higher threshold for real-time reliability
            "sampling_rate": 16000,     # Expected sample rate for audio
            "min_speech_duration_ms": 250,  # Shorter for faster response
            "min_silence_duration_ms": 150, # Shorter for quicker segmentation
            "window_size_samples": 512, # Size of each window to process - MUST be 512 for 16kHz
            "speech_pad_ms": 100,       # Less padding for faster processing
            "model_path": None,         # Path to a local model (optional)
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available()
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
            
        # Set device
        self.device = "cpu"  # Default to CPU for compatibility
        if self.config["use_cuda"] and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA for Silero VAD")
        elif self.config["use_mps"] and torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS for Silero VAD")
        else:
            logger.info("Using CPU for Silero VAD")
            
        # Initialize model
        self._initialize_model()
        logger.info(f"Silero VAD initialized on {self.device} with threshold {self.config['threshold']}")
        
    def _initialize_model(self):
        """
        Initialize the Silero VAD model with configuration parameters for quality.
        """
        if self.config["model_path"] and os.path.exists(self.config["model_path"]):
            logger.info(f"Loading Silero VAD from local file: {self.config['model_path']}")
            self.model = torch.jit.load(self.config["model_path"])
        else:
            logger.info("Loading Silero VAD from PyTorch Hub")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            
            # Get required utility functions
            self.get_speech_timestamps = self.utils[0]
            self.save_audio = self.utils[1]
            self.read_audio = self.utils[2]
            self.VADIterator = self.utils[3]
            self.collect_chunks = self.utils[4]
        
        # Ensure the window size is 512 samples for 16kHz audio (strict Silero VAD requirement)
        if self.config["sampling_rate"] == 16000 and self.config["window_size_samples"] != 512:
            logger.warning(f"Adjusted window size to 512 for 16kHz audio to meet Silero VAD requirements")
            self.config["window_size_samples"] = 512
            
        logger.info(f"Silero VAD initialized with window size: {self.config['window_size_samples']}, threshold: {self.config['threshold']}")
        print(f"✅ Silero VAD initialized with window size: {self.config['window_size_samples']}, threshold: {self.config['threshold']}")
        print(f"   Note: Silero VAD strictly requires 512 samples for 16kHz audio")
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
            
    def is_speech(self, audio_segment: np.ndarray, return_timestamps: bool = False) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Determine if the audio segment contains speech using Silero VAD.
        
        Args:
            audio_segment: Numpy array of audio (expected to be float32, mono, 16kHz)
            return_timestamps: Whether to return detailed timestamps
            
        Returns:
            Tuple of (is_speech, timestamps_if_requested)
        """
        try:
            # Convert to float32 and ensure correct shape
            audio = audio_segment.astype(np.float32)
            
            # Convert numpy array to tensor
            audio_tensor = torch.from_numpy(audio).to(self.device)
            
            # Use correct window size for the sample rate
            # Silero requires exactly 512 samples for 16kHz audio
            window_size_samples = 512  # Fixed for 16kHz audio
            
            # Get speech timestamps
            timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                threshold=self.config["threshold"],
                sampling_rate=self.config["sampling_rate"],
                min_speech_duration_ms=self.config["min_speech_duration_ms"],
                min_silence_duration_ms=self.config["min_silence_duration_ms"],
                window_size_samples=window_size_samples,  # Use correct fixed size
                speech_pad_ms=self.config["speech_pad_ms"]
            )
            
            # Determine if there's speech based on timestamps
            has_speech = len(timestamps) > 0
            
            if return_timestamps:
                return has_speech, timestamps
            else:
                return has_speech, None
                
        except Exception as e:
            logger.error(f"Error in Silero VAD processing: {e}")
            # Fall back to basic energy detection in case of error
            energy = np.sqrt(np.mean(np.square(audio_segment)))
            energy_threshold = 0.005
            is_speech_by_energy = energy > energy_threshold
            
            # Log the fallback
            logger.warning(f"Using energy fallback due to Silero error: speech={'yes' if is_speech_by_energy else 'no'}, energy={energy:.5f}")
            
            return is_speech_by_energy, None
    

            


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
    
    def validate_transcription(self, text, confidence=None):
        """
        Simple validation focused on text length and content quality.
        """
        # Skip empty or very short text (likely noise)
        if not text or len(text.strip()) < 3:
            return False
            
        # Check if text contains actual words (not just punctuation or spaces)
        if not any(c.isalpha() for c in text):
            return False
            
        # Return true for valid text
        return True
        
    def _clean_text(self, text):
        """Basic text cleaning."""
        return text.strip()


    

    


class RealTimeTranscriber:
    def __init__(self, config=None):
        # Optimized configuration for real-time performance with minimal hallucinations
        default_config = {
            "sample_rate": 16000,
            "chunk_size": 1600,        # 100ms chunks for faster real-time response
            "buffer_size": 4,         # Smaller buffer reduces latency
            "silence_threshold": 0.015, # Higher threshold to reduce hallucinations and false positives
            "min_voice_duration": 0.25, # Shorter to capture brief speech segments
            "min_silence_duration": 0.2, # Shorter silence for quicker segmentation
            "language": DEFAULT_LANGUAGE,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "num_threads": min(4, os.cpu_count() or 2),  # Fewer threads for stability
            "max_speakers": 2,  # Optimized for 2 speakers
            "speaker_similarity_threshold": 0.85,  # Higher threshold for more accurate speaker identification
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
        
        # VAD settings optimized for higher sensitivity with decent quality
        self.vad_threshold = 0.25  # More sensitive threshold (was 0.35)
        silero_config = {
            "threshold": self.vad_threshold,
            "min_speech_duration_ms": 300,   # Shorter to capture more speech (was 400ms)
            "min_silence_duration_ms": 500,  # Longer to keep more continuous speech
            "window_size_samples": 512,      # MUST be 512 for 16kHz audio (Silero requirement)
            "speech_pad_ms": 200,           # More padding to avoid cutting off speech
            "sampling_rate": self.sample_rate,
            "use_cuda": self.config.get("use_cuda", False),
            "use_mps": self.config.get("use_mps", True)
        }
        self.silero_vad = SileroVAD(silero_config)
        logger.info("Silero VAD initialized with correct window size (512 samples) for 16kHz audio")
        print(f"✅ Silero VAD initialized with window size: 512, threshold: {silero_config['threshold']}")
        print("   Note: Silero VAD strictly requires 512 samples for 16kHz audio")
        
        # Ensure MPS fallback is properly set for maximum compatibility
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
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
            # Apple Silicon MPS compatibility requires special handling
            using_mps = self.device == "mps"
            torch_device = "cpu"  # Always start on CPU for maximum compatibility
            
            # Load processor with optimizations
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with optimized settings - always use float32 for maximum compatibility
            # Do NOT use float16 as it causes dtype mismatches with the pipeline
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Always use float32 for consistent dtype
                attn_implementation="sdpa",  # Faster attention implementation
                low_cpu_mem_usage=True,     # Reduce memory usage
                use_safetensors=True        # Use safer tensor storage
            )
            
            # Store a CPU copy for inference that never goes to MPS
            self.model_cpu = self.model.to("cpu").eval()
            
            # Move main model to target device only for storage
            if using_mps:
                # Only move to MPS for storage - never for inference
                logger.info("Creating separate MPS copy of model for storage")
                self.model = self.model.to("mps")
                logger.info(f"Model copied to {self.device} device for storage")
            
            # Create ONE single CPU pipeline for maximum compatibility
            # We've removed the GPU pipeline entirely as it causes dtype issues
            logger.info("Creating CPU-only pipeline for guaranteed compatibility")
            self.whisper_pipe_cpu = pipeline(
                "automatic-speech-recognition",
                model=self.model_cpu,        # Always use CPU model for inference
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=torch.float32,   # Always use float32 on CPU
                chunk_length_s=8,            # 8s chunk for better streaming 
                stride_length_s=2,           # 2s overlap for better continuity
                batch_size=1,                # Small batch for real-time processing
                device="cpu"                 # Explicitly set CPU device
            )
            
            # Use only one pipeline for simplicity and to avoid any confusion
            # This ensures we're always using the CPU pipeline everywhere
            self.whisper_pipe = self.whisper_pipe_cpu
            
            logger.info(f"Whisper model loaded successfully on {torch_device} for inference")
            logger.info(f"Storage model device: {next(self.model.parameters()).device}")
            logger.info(f"Inference model device: {next(self.model_cpu.parameters()).device}")
            
            
            # Updated generate parameters that are compatible with Whisper v3 pipeline
            # Only use parameters that are accepted by the model
            self.generate_kwargs = {
                "task": "transcribe",
                "language": self.config.get("language", "en"),
                "temperature": 0.0
                # Removed problematic parameters:
                # compression_ratio_threshold, logprob_threshold, no_speech_threshold
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
                try:
                    from nemo.collections.asr.parts.utils.offline_clustering import (
                        SpeakerClustering
                    )

                    # Get clustering parameters from config
                    clustering_config = self.config.get("clustering", {})
                    eps = clustering_config.get("eps", 0.15)
                    max_speakers = clustering_config.get("max_speakers", 8)
                    enhanced = clustering_config.get("enhanced", True)
                    
                    # Check which parameters are supported by SpeakerClustering
                    # In newer NeMo versions, min_samples might be removed
                    import inspect
                    sig = inspect.signature(SpeakerClustering.__init__)
                    cluster_params = {}
                    if 'eps' in sig.parameters:
                        cluster_params['eps'] = eps
                    if 'max_num_speakers' in sig.parameters:
                        cluster_params['max_num_speakers'] = max_speakers
                    if 'enhanced' in sig.parameters:
                        cluster_params['enhanced'] = enhanced
                    if 'oracle_num_speakers' in sig.parameters:
                        cluster_params['oracle_num_speakers'] = False
                    if 'metric' in sig.parameters:
                        cluster_params['metric'] = 'cosine'
                    if 'min_samples' in sig.parameters:
                        cluster_params['min_samples'] = clustering_config.get("min_samples", 2)
                    
                    # Initialize clustering object for dynamic usage
                    self.clustering = SpeakerClustering(**cluster_params)
                    logger.info(f"NeMo clustering initialized with params: {cluster_params}")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize NeMo clustering with detailed params: {e}")
                    # Try a simpler initialization approach
                    try:
                        self.clustering = SpeakerClustering()
                        logger.info("NeMo clustering initialized with default parameters")
                    except Exception as e2:
                        logger.warning(f"Failed to initialize NeMo clustering with defaults: {e2}")
                
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
            logger.info("VAD processing thread started")
            
            # State variables for speech segmentation
            in_speech = False
            continuous_silence = 0
            current_segment = []
            max_segment_length = 10.0  # Max segment length in seconds
            
            while self.is_running:
                try:
                    # Try to get audio chunk from queue with timeout
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Check for non-empty audio
                    if audio_chunk is None or len(audio_chunk) == 0:
                        self.audio_queue.task_done()
                        continue
                    
                    # Check for voice activity using Silero VAD
                    has_voice = self.detect_voice_activity(audio_chunk)
                
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
                                    
                                    # Process segments with better balance between sensitivity and quality
                                    min_voice_dur = self.config.get("min_voice_duration", 0.35)  # Reduced to capture more speech segments
                                    if segment_duration >= min_voice_dur:
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
                            
                            if rms > self.config.get("silence_threshold", 0.01):
                                if not self.vad_queue.full():
                                    print(f"Long speech segment (duration: {segment_duration:.2f}s) sent for processing")
                                    self.vad_queue.put(segment_audio)
                                else:
                                    print("WARNING: VAD queue full, dropping long segment")
                            else:
                                print(f"Long speech segment has low energy, dropping")
                                
                            # Keep last second for context in next segment
                            last_samples = min(int(1.0 * self.sample_rate), len(segment_audio))
                            if len(segment_audio) > last_samples:
                                current_segment = [segment_audio[-last_samples:]]
                            else:
                                current_segment = [segment_audio]
                    
                    self.audio_queue.task_done()
                    
                except queue.Empty:
                    # No audio available, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing audio for VAD: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            print(f"VAD processing error: {e}")
            
    def detect_voice_activity(self, audio_chunk, threshold=None):
        """
        Voice Activity Detection using Silero VAD and energy heuristics.
        Optimized for higher sensitivity with decent quality.
        
        Args:
            audio_chunk: Audio numpy array
            threshold: Not used with Silero VAD (kept for API compatibility)
            
        Returns:
            bool: True if voice detected, False otherwise
        """
        # Calculate energy regardless of approach
        energy = np.sqrt(np.mean(np.square(audio_chunk)))
        
        # Energy-based pre-filter - IMMEDIATELY ACCEPT higher energy as speech
        # Use this for quick detection of obvious speech sounds
        if energy > 0.02:  # More sensitive threshold (was 0.04)
            print(f"🔊 HIGH ENERGY SPEECH DETECTED: {energy:.5f}, Length: {len(audio_chunk)}")
            return True
        
        # Skip if chunk is too short
        if len(audio_chunk) < 512:  # Silero requires at least 512 samples at 16kHz
            # More sensitive energy detection for small chunks
            energy_threshold = 0.005  # More sensitive (was 0.01)
            is_speech = energy > energy_threshold
            if is_speech:
                print(f"🎤 Small chunk speech - Energy: {energy:.5f}")
            return is_speech
        
        try:
            # Skip very quiet audio
            if energy < 0.0005:  # More sensitive (was 0.001)
                return False
                
            # Pad audio to make it a multiple of 512 if needed
            remainder = len(audio_chunk) % 512
            if remainder > 0:
                padding = np.zeros(512 - remainder, dtype=np.float32)
                usable_audio = np.concatenate([audio_chunk, padding])
            else:
                usable_audio = audio_chunk
            
            # Use Silero VAD - requires exactly 512 sample chunks for 16kHz
            has_speech, _ = self.silero_vad.is_speech(usable_audio)
            
            # SPECIAL CASE: For moderately strong signals, trust energy over Silero
            # More aggressive energy overrides
            if not has_speech and energy > 0.01:  # More sensitive (was 0.02)
                print(f"🔊 ENERGY OVERRIDE: {energy:.5f}, Length: {len(audio_chunk)}")
                return True
            
            # Log the result
            if has_speech:
                print(f"🎤 SPEECH detected - Energy: {energy:.5f}, Length: {len(audio_chunk)}")
            elif energy > 0.005:  # More sensitive (was 0.01)
                print(f"🔇 No speech - Energy: {energy:.5f}, Length: {len(audio_chunk)}")
                
            return has_speech
            
        except Exception as e:
            # Fallback to simple energy check if Silero fails
            logger.warning(f"Silero VAD error: {e} - falling back to energy detection")
            
            # Calculate energy - root mean square (RMS) amplitude
            energy = np.sqrt(np.mean(np.square(audio_chunk)))
            energy_threshold = 0.005  # Higher threshold than original as fallback
            
            is_speech = energy > energy_threshold
            print(f"⚠️ Using energy fallback - Energy: {energy:.5f}, Speech: {'yes' if is_speech else 'no'}")
            
            return is_speech
    
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
        # Convert audio to proper format - ensure we're working with numpy array
        if isinstance(audio_segment, torch.Tensor):
            audio_np = audio_segment.detach().cpu().numpy().astype(np.float32)
        else:
            audio_np = audio_segment.astype(np.float32)
        
        # Get config parameters
        max_speakers = self.config.get("max_speakers", 3)
        similarity_threshold = self.config.get("speaker_similarity_threshold", 0.7)
        
        # Compute speaker embedding using NeMo's TitaNet
        if not self.use_nemo:
            return "Speaker_1", None
            
        try:
            # TitaNet needs a temporary WAV file for processing - this is the most reliable method
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                # Normalize audio before writing to file
                if np.max(np.abs(audio_np)) > 0:
                    audio_np = audio_np / np.max(np.abs(audio_np))
                
                # Write normalized audio to temp file with proper sample rate
                sf.write(tmp_file.name, audio_np, self.sample_rate)
                
                # Get the embedding using the file path (most reliable method for MPS compatibility)
                with torch.inference_mode():  # Stricter than no_grad
                    try:
                        # Always compute embeddings on CPU for reliability
                        original_device = next(self.speaker_model.parameters()).device
                        cpu_model = self.speaker_model.to('cpu')
                        
                        # Get embedding from file
                        embedding = cpu_model.get_embedding(tmp_file.name)
                        
                        # Always ensure embedding is on CPU and properly shaped
                        embedding_cpu = embedding.cpu().float()
                        
                        # Move model back to original device for storage
                        self.speaker_model = self.speaker_model.to(original_device)
                    except Exception as e:
                        logger.error(f"Error getting speaker embedding: {e}")
                        # If everything fails, return default speaker
                        return "Speaker_1", None
            
            # Initialize for comparison
            max_similarity = -1
            closest_speaker = None
            
            # If no speakers yet, create first speaker
            if len(self.speakers) == 0:
                new_id = "Speaker_1"
                # Create embedding as 2D for sklearn compatibility
                embedding_shaped = embedding_cpu.unsqueeze(0) if embedding_cpu.dim() == 1 else embedding_cpu
                self.speakers[new_id] = SpeakerProfile(id=new_id, embedding=embedding_shaped)
                return new_id, embedding_shaped
            
            # Compare with existing speakers
            similarities = []
            for speaker_id, profile in self.speakers.items():
                if profile.embedding is not None:
                    try:
                        # Ensure stored embedding is on CPU and properly shaped
                        stored_cpu = profile.embedding.cpu().float()
                        
                        # Ensure both are 2D for comparison
                        emb1 = embedding_cpu.unsqueeze(0) if embedding_cpu.dim() == 1 else embedding_cpu
                        emb2 = stored_cpu.unsqueeze(0) if stored_cpu.dim() == 1 else stored_cpu
                        
                        # Compute similarity with torch for consistency
                        with torch.inference_mode():
                            sim = F.cosine_similarity(emb1, emb2).item()
                        
                        similarities.append((speaker_id, sim))
                        logger.debug(f"Similarity with {speaker_id}: {sim:.3f}")
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
                    self.speakers[best_match_id].update_embedding(embedding_cpu, 0.3)
                    logger.info(f"Speaker {best_match_id} matched with similarity {best_sim:.3f}")
                    return best_match_id, embedding_cpu
                
                # If we've reached max speakers, return the most similar one anyway
                if len(self.speakers) >= max_speakers:
                    # Update speaker's embedding slightly
                    self.speakers[best_match_id].update_embedding(embedding_cpu, 0.1)
                    logger.info(f"Max speakers reached. Using {best_match_id} (similarity: {best_sim:.3f})")
                    return best_match_id, embedding_cpu
                    
                # Create a new speaker if below max limit
                new_id = f"Speaker_{len(self.speakers) + 1}"
                # Ensure embedding is properly shaped for storage
                embedding_shaped = embedding_cpu.unsqueeze(0) if embedding_cpu.dim() == 1 else embedding_cpu
                self.speakers[new_id] = SpeakerProfile(id=new_id, embedding=embedding_shaped)
                logger.info(f"New speaker detected: {new_id}")
                return new_id, embedding_shaped
            
            # Fallback to the most similar speaker if we have similarities but no good match
            if similarities:
                best_match_id = similarities[0][0]
                # Update speaker's embedding minimally
                self.speakers[best_match_id].update_embedding(embedding_cpu, 0.1)
                logger.info(f"Using most similar speaker {best_match_id} as fallback")
                return best_match_id, embedding_cpu
            
            # Ultimate fallback - should never reach here if speaker detection is working
            default_id = "Speaker_1"
            if default_id not in self.speakers:
                # Create first speaker if it doesn't exist
                embedding_shaped = embedding_cpu.unsqueeze(0) if embedding_cpu.dim() == 1 else embedding_cpu
                self.speakers[default_id] = SpeakerProfile(id=default_id, embedding=embedding_shaped)
            return default_id, embedding_cpu
                
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            # Return a default speaker ID
            return "Speaker_1", None
    
    def transcription_thread(self):
        """Transcription processing thread."""
        logger.info("Transcription thread started")
        print("\n🚀 Transcription thread started - Ready to transcribe speech")
        segment_counter = 0
        success_counter = 0
        failure_counter = 0
        last_stats_time = time.time()
        
        while self.is_running:
            # Get diarized segment
            try:
                segment = self.diarization_queue.get(timeout=0.5)
                
                # Show queue sizes periodically
                now = time.time()
                if now - last_stats_time > 10.0:  # Every 10 seconds
                    print(f"\n📊 QUEUE STATUS: Audio: {self.audio_queue.qsize()}/{self.audio_queue.maxsize}, "
                          f"VAD: {self.vad_queue.qsize()}/{self.vad_queue.maxsize}, "
                          f"Diarization: {self.diarization_queue.qsize()}/{self.diarization_queue.maxsize}, "
                          f"Success rate: {success_counter}/{segment_counter} ({success_counter/max(1,segment_counter)*100:.1f}%)")
                    last_stats_time = now
                    
            except queue.Empty:
                continue

            try:
                segment_counter += 1
                audio_duration = len(segment['audio']) / self.sample_rate if 'audio' in segment else 0
                audio_energy = np.sqrt(np.mean(np.square(segment['audio']))) if 'audio' in segment else 0
                
                print(f"\n🔊 Processing segment #{segment_counter} | Speaker: {segment['speaker_id']} | "
                      f"Duration: {audio_duration:.2f}s | Energy: {audio_energy:.5f}")
                
                # Run Whisper transcription
                start_time = time.time()
                transcription = self.transcribe_audio(segment["audio"])
                trans_time = time.time() - start_time
                
                if transcription:
                    success_counter += 1
                    print(f"✅ Transcription completed in {trans_time:.2f}s: \"{transcription}\"")
                else:
                    failure_counter += 1
                    print(f"❌ No valid transcription after {trans_time:.2f}s")
                    continue

                # Skip if no valid transcription was found
                if not transcription or not transcription.strip():
                    print("❌ Empty transcription result")
                    continue

                # Exact match duplicate detection - only check exact string matches
                # to avoid over-filtering similar content
                if transcription in self.seen_texts or (
                    self.transcript_history and transcription == self.transcript_history[-1]["text"]
                ):
                    print(f"🔄 Duplicate transcription - skipping: '{transcription}'")
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
            
            # Simple approach: Always use our dedicated CPU pipeline
            try:
                # Ensure we're using float32 numpy array for CPU inference
                audio_float32 = audio_np.astype(np.float32)
                
                # Run inference on CPU with numpy array using our dedicated CPU pipeline
                with torch.inference_mode():  # Even stricter than no_grad for inference
                    # Use .copy() to ensure we have a completely fresh array
                    result = self.whisper_pipe_cpu(
                        audio_float32.copy(),  # Use a copy to prevent dtype contamination
                        generate_kwargs=self.generate_kwargs,
                        return_timestamps=False
                    )
                
            except Exception as e:
                logger.error(f"Transcription error with CPU pipeline: {e}")
                # Fallback: Direct processor + model approach with basic parameters only
                try:
                    # Use dedicated CPU model and processor directly
                    with torch.inference_mode():
                        # Use a fresh copy of the audio for processing to avoid any dtype contamination
                        # Ensure it's float32 only
                        clean_audio = audio_np.copy().astype(np.float32)
                        
                        # Process audio directly 
                        input_features = self.processor(
                            clean_audio, 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).input_features
                        
                        # Ensure input features are correct dtype
                        input_features = input_features.to(dtype=torch.float32)
                        
                        # Generate transcription with CPU model - use minimal parameters
                        # to avoid conflicts with what the model supports
                        language = self.config.get("language", "en")
                        outputs = self.model_cpu.generate(
                            input_features,
                            task="transcribe",
                            language=language
                        )
                        
                        # Decode outputs
                        text = self.processor.batch_decode(
                            outputs, 
                            skip_special_tokens=True
                        )[0].strip()
                        
                        # Format result like pipeline output
                        result = {"text": text}
                        logger.info(f"Fallback transcription succeeded: '{text}'")
                        
                        
                except Exception as e2:
                    logger.error(f"Complete transcription failure: {e2}")
                    return None

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
        "chunk_size": 1600,                 # Larger chunks for better context (500ms)
        "min_voice_duration": 0.7,          # Minimum speech segment for quality
        "min_silence_duration": 0.2,        # Better segmentation between utterances
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