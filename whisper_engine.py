# whisper_engine.py - Pure Whisper Backend Engine (Minimal Logging)

import whisper
import torch
import time
import numpy as np
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import librosa

# Suppress all unnecessary output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

class WhisperEngine:
    """
    Local Whisper Engine for Speech-to-Text conversion
    Pure backend implementation with minimal logging
    """
    
    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """
        Initialize Whisper Engine
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computation"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return device
    
    def _load_model(self):
        """Load Whisper model silently"""
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def _load_audio_with_librosa(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio using librosa"""
        try:
            audio_data, sample_rate = librosa.load(
                str(audio_path), 
                sr=16000,
                mono=True
            )
            
            # Ensure audio is float32 and in correct range
            audio_data = audio_data.astype(np.float32)
            
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
        }
    
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
        
        # Validate input
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Load audio
            audio_data = self._load_audio_with_librosa(audio_path)
            audio_duration = len(audio_data) / 16000
            
            # Transcribe with suppressed output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = self.model.transcribe(
                    audio_data,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                    verbose=False
                )
            
            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            # Enhanced result
            enhanced_result = {
                "text": result["text"].strip(),
                "language": result["language"],
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
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'confidence': word_info.get('probability', 1.0)
                    })
        
        return words