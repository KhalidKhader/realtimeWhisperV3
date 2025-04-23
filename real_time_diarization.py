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

class RealTimeTranscriber:
    def __init__(self, config=None):
        # Default configuration thresholds tuned for clarity
        default_config = {
            "sample_rate": 16000,
            "chunk_size": 4000,        # 250ms chunks
            "buffer_size": 30,         # 30s context buffer
            "silence_threshold": 0.003, # VAD energy threshold (very sensitive to capture all speech)
            "calibration_duration": 2.0, # seconds to record ambient noise at startup
            "calibration_factor": 1.2,   # multiplier for ambient noise RMS to set silence_threshold
            "min_voice_duration": 0.1,   # capture even very short utterances
            "min_silence_duration": 0.05, # split on very short silences for better segmentation
            "no_speech_threshold": 0.85,
            "language": DEFAULT_LANGUAGE,
            "use_cuda": torch.cuda.is_available(),
            "num_threads": min(8, os.cpu_count() or 2),  # use more threads if available
            "simple_diarization": False,  # disable simple alternating speaker diarization
            "speaker_similarity_threshold": 0.45,  # Lower threshold to more aggressively merge similar speakers
            "max_speakers": 2,  # Limit to 2 speakers for medical conversations
            "capture_all_speech": True,  # capture all voices in main run
            "blocked_phrases": [
                "thank you for watching",
                "takk for watching",
                "thanks for watching",
                "sous-titrage société radio-canada",
                "takk for att du så på",
                "terima kasih telah menonton",
                "subtitled by",
                "captions",
                "subtitles by",
                "tchau",
                "bye-bye",
                "obrigado",
                "مرحباً",
                "شكراً",
                "e aí",
                "gracias",
                "حسنا",
                "لنبدأ",
                "بالتوصيل",
                "بالتصوير"
            ],
            # NeMo clustering parameters
            "clustering": {
                "min_samples": 2,      # Lower min samples to merge clusters more easily
                "eps": 0.25,           # Higher eps for more aggressive clustering (less discriminative)
                "max_speakers": 2,     # Maximum number of speakers to detect
                "window_size": 120,    # Longer context window for better clustering
                "enhanced": True,      # Use NeMo's enhanced clustering
                "fallback_threshold": 0.45  # Lower threshold for more aggressive merging
            },
            "speech_language": {
                "min_confidence": 0.75,  # Minimum confidence in language detection
                "non_latin_ratio": 0.1,  # Maximum allowed non-Latin character ratio
                "target_lang_confidence": 0.4  # Minimum confidence in target language
            }
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
        
        # Mark model initialization as complete
        self.initialization_complete = True
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
            
            # Fix the forced_decoder_ids warning by setting empty list
            self.empty_decoder_ids = []
            
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
            
            # Initialize clustering parameters for diarization
            try:
                # Import NeMo clustering and diarization components
                import nemo.collections.asr.parts.utils.speaker_utils as speaker_utils
                from nemo.collections.asr.parts.utils.offline_clustering import (
                    SpeakerClustering, 
                    get_argmin_mat,
                    split_input_data
                )
                from nemo.collections.asr.parts.utils.speaker_utils import (
                    get_timestamps_from_manifest, 
                    embedding_normalize
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
        
        speech_segments = []
        continuous_silence = 0
        in_speech = False
        current_segment = []
        
        # Enhanced parameters for better voice capture
        max_segment_length = 10.0  # maximum segment length in seconds before forced processing
        min_segment_energy = 0.001  # minimum energy to consider for processing
        
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
                    if segment_audio.size == 0:
                        continue  # Skip empty segments 
                    segment_duration = len(segment_audio) / self.sample_rate
                    
                    # Process if segment is getting long (>5 seconds)
                    if segment_duration > max_segment_length:
                        # Check if segment has minimum energy to be worth processing
                        rms = np.sqrt(np.mean(segment_audio**2)) if segment_audio.size > 0 else 0
                        if rms < min_segment_energy:
                            logger.debug(f"Skipping low-energy segment: {rms:.6f}")
                            current_segment = []
                            continue
                        
                        if not self.vad_queue.full():
                            try:
                                self.vad_queue.put(segment_audio, block=False)
                            except queue.Full:
                                logger.warning("VAD queue full, skipping segment")
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
        """Identify speaker using NeMo speaker embeddings and clustering."""
        try:
            # Get max speakers limit
            max_speakers = self.config.get("max_speakers", 2)
            
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
                
                # Record current timestamp for windowing
                current_time = time.time()
                
                # Add the new embedding to our buffer
                self.speaker_embeddings_buffer.append((embedding, current_time))
                
                # Maintain buffer size by removing older embeddings
                window_size = self.config.get("clustering", {}).get("window_size", 120)
                self.speaker_embeddings_buffer = [
                    (emb, ts) for emb, ts in self.speaker_embeddings_buffer 
                    if current_time - ts < window_size
                ]
                
                # If less than max_speakers exist, and it's the first time, create first speaker
                if len(self.speakers) == 0:
                    return "Speaker_1", embedding
                    
                # If we already have speaker profiles, calculate similarities
                if self.speakers:
                    similarities = []
                    for speaker_id, profile in self.speakers.items():
                        if profile.embedding is not None:
                            sim = cosine_similarity([embedding], [profile.embedding])[0][0]
                            similarities.append((speaker_id, sim))
                    
                    # If we found any similarities
                    if similarities:
                        # Sort by similarity (highest first)
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        best_match_id, best_sim = similarities[0]
                        threshold = self.config.get("speaker_similarity_threshold", 0.45)
                        
                        # If found a good match, return that speaker
                        if best_sim > threshold:
                            # Update speaker's embedding with this new one for better tracking
                            self.speakers[best_match_id].update_embedding(embedding, 0.5)
                            return best_match_id, embedding
                        
                        # If we've reached max speakers, return the most similar one anyway
                        if len(self.speakers) >= max_speakers:
                            # Update speaker's embedding with this new one for better tracking
                            self.speakers[best_match_id].update_embedding(embedding, 0.2)
                            return best_match_id, embedding
                    
                # If we haven't hit max speakers yet, create a new one
                if len(self.speakers) < max_speakers:
                    new_id = f"Speaker_{len(self.speakers) + 1}"
                    return new_id, embedding
                    
                # Fallback: return the first speaker if we somehow got here
                return "Speaker_1", embedding
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
                        # Convert to our naming convention, but enforce max_speakers
                        numeric_id = speaker_id.replace("SPEAKER_", "").replace("speaker_", "")
                        try:
                            speaker_num = int(numeric_id) % max_speakers + 1
                        except:
                            speaker_num = len(self.speakers) % max_speakers + 1
                        speaker_id = f"Speaker_{speaker_num}"
                    else:
                        if len(self.speakers) == 0:
                            speaker_id = "Speaker_1"
                        else:
                            # Alternate between existing speakers
                            speaker_num = (len(self.transcript_history) % max_speakers) + 1
                            speaker_id = f"Speaker_{speaker_num}"
                    
                    # No embedding in PyAnnotate fallback
                    return speaker_id, None
                
        except Exception as e:
            logger.error(f"Speaker identification error: {e}")
            # Fallback to alternating speakers
            if not self.transcript_history:
                return "Speaker_1", None
            else:
                # Get the most recent speaker and alternate
                last_speaker = self.transcript_history[-1]["speaker_id"]
                if last_speaker == "Speaker_1":
                    return "Speaker_2", None
                else:
                    return "Speaker_1", None
    
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

                # Additional check for partial matches of phrases that often appear in outro segments
                close_match_phrases = ["thank", "thanks", "watching", "takk", "bye", "subtitle"]
                if any(phrase.lower() in transcription.lower() for phrase in close_match_phrases):
                    logger.info(f"Likely outro/intro phrase detected, skipping: {transcription}")
                    continue

                # Suppress seen or duplicate transcripts
                if transcription in self.seen_texts or (
                    self.transcript_history and transcription == self.transcript_history[-1]["text"]
                ):
                    continue
                self.seen_texts.add(transcription)

                # Use speaker ID directly without role classification
                speaker_id = segment["speaker_id"]

                # Build and queue result
                result = {
                    "speaker_id": speaker_id,
                    "role": speaker_id,  # Use speaker_id as the role
                    "text": transcription,
                    "timestamp": segment["timestamp"]
                }
                
                if speaker_id not in self.speakers:
                    self.speakers[speaker_id] = SpeakerProfile(
                        id=speaker_id,
                        embedding=segment["embedding"]
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

            # Create proper attention mask to fix the warning
            # First extract features properly with attention mask
            inputs = self.processor.feature_extractor(
                audio_np, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                return_attention_mask=True
            )
            # Use proper attention mask from processor
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
            
            # Convert to list format that the pipeline expects
            if attention_mask is not None:
                # Create an attention mask where valid tokens have 1.0 and padding has 0.0
                audio_np = {"array": audio_np, "sampling_rate": self.sample_rate, "attention_mask": attention_mask}

            result = self.whisper_pipe(
                audio_np,
                generate_kwargs={
                    "task": "transcribe",
                    "temperature": 0.0,
                    "forced_decoder_ids": self.empty_decoder_ids,  # explicitly set empty to fix warning
                    "compression_ratio_threshold": 1.5,  # more repetition filtering
                    "logprob_threshold": 0.0,            # require non-negative logprob
                    "no_speech_threshold": self.config["no_speech_threshold"],
                    "num_beams": 3,                      # beam search for accuracy
                    "do_sample": False,                   # disable sampling for consistent results
                    "return_legacy_cache": True,          # address cache warning
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

            # Language confidence check - replace blocklist with confidence check
            if "text" in result:
                text = result["text"].strip()
                # Use advanced language detection to check for confidence
                if "language_probability" in result:
                    lang_confidence = result["language_probability"]
                    # Only keep high-confidence transcriptions
                    if lang_confidence < 0.75:  # 75% confidence threshold
                        logger.info(f"Ignoring low confidence ({lang_confidence:.2f}) transcript: {text}")
                        return None

                # Foreign script detection
                non_latin_ratio = sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
                if non_latin_ratio > 0.1:  # More than 10% non-Latin characters
                    logger.info(f"Ignoring transcript with non-Latin characters: {text}")
                    return None
                
                # Check for common multilingual patterns and code-switching
                # This approach is more flexible than a blocklist
                try:
                    # Try to detect potential code-switching
                    detected_langs = detect_langs(text)
                    
                    # If the top language isn't our target language
                    top_lang = detected_langs[0]
                    if top_lang.lang != self.config["language"] and top_lang.prob > 0.6:
                        logger.info(f"Ignoring code-switched text. Detected {top_lang.lang} with prob {top_lang.prob}: {text}")
                        return None
                    
                    # If we have high confidence in multiple languages, it's likely code-switching
                    if len(detected_langs) > 1 and detected_langs[1].prob > 0.3:
                        logger.info(f"Detected likely code-switching: {detected_langs}")
                        
                        # If our target language is strong enough, keep it
                        target_lang = next((l for l in detected_langs if l.lang == self.config["language"]), None)
                        if not target_lang or target_lang.prob < 0.4:
                            logger.info(f"Insufficient confidence in target language: {text}")
                            return None
                except Exception as e:
                    # If language detection fails, fall back to content-based checks
                    logger.debug(f"Language detection failed: {e}")
                
                # Check for likely misrecognized medical terms
                medical_corrections = {
                    r'\ban egg\b': 'an ECG',
                    r'\begg\b': 'ECG',
                    r'\bxenoblade\b': 'Clopidogrel',
                    r'\baesir\b': 'ACE inhibitor',
                    r'\baveda\b': 'a beta blocker',
                    r'\baspirator\b': 'aspirator',
                    r'\bstark\b': 'start',
                    r'\beast cardiogram\b': 'electrocardiogram',
                    r'\bgamingcardio\b': 'angiogram'
                }
                
                for error, correction in medical_corrections.items():
                    text = re.sub(error, correction, text, flags=re.IGNORECASE)
                
                # Validate sentence structure - filter incomplete fragments
                # Check if the text is likely to be a complete sentence or thought
                def is_complete_sentence(s):
                    # Very short utterances are often incomplete
                    if len(s.split()) < 3:
                        return False
                        
                    # Check for ending with prepositions or articles - likely incomplete
                    ending_words = ['to', 'the', 'a', 'an', 'in', 'on', 'at', 'with', 'by', 'for', 'and', 'or', 'but']
                    last_word = s.split()[-1].lower().strip('.,?!')
                    if last_word in ending_words:
                        return False
                        
                    # Check if sentence has at least one complete clause structure
                    # (simplistic implementation - counts verbs)
                    has_subject = any(word.lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they'] for word in s.split())
                    common_verbs = ['am', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 
                                   'can', 'could', 'will', 'would', 'see', 'feel', 'think', 'know', 'tell', 'go', 'come']
                    has_verb = any(word.lower() in common_verbs for word in s.split())
                    
                    # Either has subject+verb structure or ends with punctuation
                    return (has_subject and has_verb) or s.strip().endswith(('.', '?', '!'))
                
                # If text appears to be an incomplete fragment, check if we should discard it
                if not is_complete_sentence(text):
                    # Only discard if it's short and looks very incomplete
                    if len(text.split()) < 8 and not any(x in text.lower() for x in ['doctor', 'patient', 'cardiac', 'heart']):
                        logger.info(f"Skipping likely incomplete sentence: {text}")
                        return None
                
                # More aggressive repetition removal - handles phrases not just single words
                # First handle extreme word repetition
                single_word_pattern = r'(\b\w+\b)(\s+\1){1,}'
                text = re.sub(single_word_pattern, r'\1', text)
                
                # Then handle repeated phrases (2+ words)
                for phrase_len in range(5, 1, -1):  # Try phrases of length 5,4,3,2 words
                    # Look for repeated phrases of this length
                    words = text.split()
                    if len(words) < phrase_len * 2:
                        continue
                        
                    i = 0
                    while i <= len(words) - phrase_len * 2:
                        phrase1 = ' '.join(words[i:i+phrase_len])
                        phrase2 = ' '.join(words[i+phrase_len:i+phrase_len*2])
                        
                        if phrase1.lower() == phrase2.lower():
                            # Remove the repetition
                            words = words[:i+phrase_len] + words[i+phrase_len*2:]
                        else:
                            i += 1
                    
                    text = ' '.join(words)
                
                return text
            return None
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def output_thread(self):
        """Handle output formatting and display."""
        logger.info("Output thread started")
        
        try:
            while self.is_running:
                try:
                    result = self.output_queue.get(timeout=0.5)
                    
                    # Format output with speaker ID instead of role
                    speaker_label = result["speaker_id"]
                    text = result["text"]
                    timestamp = result["timestamp"] - self.start_time
                    min_sec = divmod(int(timestamp), 60)
                    
                    # Print with timestamp
                    print(f"[{min_sec[0]:02d}:{min_sec[1]:02d}] {speaker_label}: {text}")
                    
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
        if not text:
            return ""
        
        # Apply existing post-processing rules
        # ... (existing code) ...
        
        # Apply deduplication to remove repeated phrases
        text = self.remove_repeated_phrases(text)
        
        # Clean up repetitive answers
        text = self.clean_repetitive_answers(text)
        
        return text

if __name__ == "__main__":
    # Configuration (best-practice thresholds)
    config = {
        "language": DEFAULT_LANGUAGE,
        "chunk_size": 4000,
        "sample_rate": 16000,
        "use_cuda": torch.cuda.is_available(),
        "num_threads": min(8, os.cpu_count() or 2),  # use more threads if available
        "silence_threshold": 0.003, # VAD energy threshold (very sensitive to capture all speech)
        "min_voice_duration": 0.1,   # capture even very short utterances
        "min_silence_duration": 0.05, # split on very short silences for better segmentation
        "no_speech_threshold": 0.85,
        # NeMo clustering parameters
        "clustering": {
            "min_samples": 3,      # Require more samples for a cluster
            "eps": 0.12,           # Tighter clustering threshold
            "max_speakers": 8,     # Maximum number of speakers to detect
            "window_size": 60,     # Longer context window for better clustering
            "enhanced": True,      # Use NeMo's enhanced clustering
            "fallback_threshold": 0.70  # Higher threshold for better discrimination
        },
        "speech_language": {
            "min_confidence": 0.75,  # Minimum confidence in language detection
            "non_latin_ratio": 0.1,  # Maximum allowed non-Latin character ratio
            "target_lang_confidence": 0.4  # Minimum confidence in target language
        }
    }
    
    # Create and start the transcriber
    transcriber = RealTimeTranscriber(config)
    transcriber.start()