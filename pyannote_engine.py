# pyannote_engine.py - Pure PyAnnote Backend Engine (Minimal Logging)

from pyannote.audio import Pipeline
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import librosa
import soundfile as sf
import tempfile
import os
import warnings
import logging

# Suppress all unnecessary output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)

class PyAnnoteEngine:
    """
    PyAnnote Speaker Diarization Engine
    Pure backend implementation with minimal logging
    """
    
    def __init__(self, device: str = "auto", use_auth_token: bool = True):
        """
        Initialize PyAnnote Engine
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            use_auth_token: Whether to use HuggingFace auth token
        """
        self.device = self._setup_device(device)
        self.pipeline = None
        self.use_auth_token = use_auth_token
        self.temp_files = []
        self._load_pipeline()
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computation"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return device
    
    def _load_pipeline(self):
        """Load PyAnnote diarization pipeline silently"""
        try:
            if self.use_auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True
                )
            else:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PyAnnote pipeline: {e}")
    
    def _preprocess_audio_for_pyannote(self, audio_path: Path) -> str:
        """Preprocess audio to ensure compatibility with PyAnnote"""
        try:
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                str(audio_path), 
                sr=16000,
                mono=True,
                res_type='kaiser_fast'
            )
            
            # Ensure audio length constraints
            min_duration = 1.0
            max_duration = 3600.0
            
            duration = len(audio_data) / 16000
            
            if duration < min_duration:
                target_length = int(min_duration * 16000)
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
            
            elif duration > max_duration:
                target_length = int(max_duration * 16000)
                audio_data = audio_data[:target_length]
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Create temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='pyannote_')
            os.close(temp_fd)
            
            # Save preprocessed audio
            sf.write(temp_path, audio_data, 16000, subtype='PCM_16')
            
            # Track for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
            
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def diarize_audio(
        self, 
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Fixed number of speakers (None for auto-detection)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Dictionary with diarization results
        """
        
        # Validate input
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        processed_audio_path = None
        
        try:
            # Preprocess audio
            processed_audio_path = self._preprocess_audio_for_pyannote(audio_path)
            
            # Perform diarization with suppressed output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                try:
                    if num_speakers:
                        diarization = self.pipeline(
                            processed_audio_path,
                            num_speakers=num_speakers
                        )
                    else:
                        diarization = self.pipeline(
                            processed_audio_path,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                except RuntimeError as e:
                    if "tensor" in str(e).lower() and "size" in str(e).lower():
                        # Fallback: try with fixed 2 speakers
                        if not num_speakers:
                            diarization = self.pipeline(
                                processed_audio_path,
                                num_speakers=2
                            )
                        else:
                            raise
                    else:
                        raise
            
            # Process results
            segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })
                speakers.add(speaker)
            
            # Calculate statistics
            total_duration = max([seg['end'] for seg in segments]) if segments else 0
            speaker_stats = {}
            
            for speaker in speakers:
                speaker_segments = [seg for seg in segments if seg['speaker'] == speaker]
                speaker_duration = sum([seg['duration'] for seg in speaker_segments])
                speaker_stats[speaker] = {
                    'segments': len(speaker_segments),
                    'total_duration': speaker_duration,
                    'percentage': (speaker_duration / total_duration * 100) if total_duration > 0 else 0
                }
            
            results = {
                'segments': segments,
                'speakers': list(speakers),
                'num_speakers': len(speakers),
                'total_duration': total_duration,
                'speaker_stats': speaker_stats,
                'metadata': {
                    'file_name': audio_path.name,
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers,
                    'num_speakers_detected': len(speakers),
                    'preprocessing_applied': True
                }
            }
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")
        finally:
            # Always clean up
            self._cleanup_temp_files()
    
    def get_speaker_timeline(self, results: Dict) -> List[Tuple[float, float, str]]:
        """
        Get simple speaker timeline
        
        Args:
            results: Results from diarize_audio()
            
        Returns:
            List of (start_time, end_time, speaker) tuples
        """
        timeline = []
        for segment in results['segments']:
            timeline.append((
                segment['start'],
                segment['end'],
                segment['speaker']
            ))
        
        return sorted(timeline)
    
    def get_speaker_stats(self, results: Dict) -> Dict:
        """Get formatted speaker statistics"""
        stats = {
            'total_duration': results['total_duration'],
            'num_speakers': results['num_speakers'],
            'total_segments': len(results['segments']),
            'speaker_breakdown': results['speaker_stats']
        }
        return stats