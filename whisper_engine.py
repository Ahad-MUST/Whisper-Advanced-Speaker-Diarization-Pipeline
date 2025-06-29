# whisper_engine.py - Optimized Whisper Backend Engine

import whisper
import torch
import time
import numpy as np
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import librosa

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

class WhisperEngine:
    """
    Optimized Local Whisper Engine for Speech-to-Text conversion
    """
    
    TARGET_SAMPLE_RATE = 16000
    
    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """
        Initialize Whisper Engine
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computation"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print("Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio efficiently"""
        try:
            # Load audio with librosa - optimized for Whisper
            audio_data, _ = librosa.load(
                str(audio_path), 
                sr=self.TARGET_SAMPLE_RATE,
                mono=True,
                res_type='kaiser_fast'  # Faster resampling
            )
            
            # Ensure audio is float32 and normalized
            audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            return audio_data
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            "model_size": self.model_size,
            "device": self.device,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            })
        
        return info
    
    def transcribe_audio(
        self, 
        audio_path: Union[str, Path], 
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file to text with timestamps
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'de', 'en') or None for auto-detection
            task: 'transcribe' or 'translate'
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Load audio
            audio_data = self._load_audio(audio_path)
            audio_duration = len(audio_data) / self.TARGET_SAMPLE_RATE
            
            print(f"Transcribing audio: {audio_duration:.1f}s")
            
            # Transcribe with error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = self.model.transcribe(
                    audio_data,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    verbose=False
                )
            
            # Calculate performance metrics
            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            print(f"Transcription complete: {processing_time:.1f}s ({speed_ratio:.1f}x real-time)")
            
            # Create enhanced result with error handling
            enhanced_result = {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "metadata": {
                    "file_name": audio_path.name,
                    "file_size_mb": audio_path.stat().st_size / 1e6,
                    "audio_duration_seconds": audio_duration,
                    "processing_time_seconds": processing_time,
                    "speed_ratio": speed_ratio,
                    "model_size": self.model_size,
                    "device": self.device,
                    "word_timestamps": word_timestamps,
                    "task": task
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def get_word_level_timestamps(self, results: Dict) -> List[Dict]:
        """
        Extract word-level timestamps from results
        
        Returns:
            List of dictionaries with word, start, end, confidence
        """
        words = []
        
        for segment in results.get('segments', []):
            segment_words = segment.get('words', [])
            for word_info in segment_words:
                try:
                    words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0),
                        'confidence': word_info.get('probability', 1.0)
                    })
                except (KeyError, TypeError):
                    continue  # Skip malformed word entries
        
        return words
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()