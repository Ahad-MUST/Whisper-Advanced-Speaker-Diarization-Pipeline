# whisperx_engine.py - WhisperX Engine with Forced Alignment and Built-in Diarization (FIXED)

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
import logging
import time
import tempfile
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

class WhisperXEngine:
    """
    WhisperX Engine - Enhanced Whisper with Forced Alignment (FIXED VERSION)
    
    Features:
    - Forced alignment for precise word-level timestamps
    - Built-in speaker diarization capabilities
    - Enhanced transcription accuracy
    - Multiple language support with optimal alignment
    - Compatible with actual WhisperX API
    
    WhisperX provides superior word-level timing compared to standard Whisper
    and includes speaker diarization capabilities for complete audio analysis.
    """
    
    def __init__(
        self, 
        model_size: str = "large-v3", 
        device: str = "auto",
        compute_type: str = "float16",
        enable_diarization: bool = True,
        batch_size: int = 1
    ):
        """
        Initialize WhisperX Engine with correct API usage
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large', 'large-v3')
            device: Device to use ('auto', 'cuda', 'cpu')
            compute_type: Computation precision ('float16', 'float32', 'int8')
            enable_diarization: Enable built-in speaker diarization
            batch_size: Batch size for faster processing
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.compute_type = compute_type
        self.enable_diarization = enable_diarization
        self.batch_size = batch_size
        
        # WhisperX models
        self.whisperx_model = None
        self.alignment_models = {}
        self.diarization_pipeline = None
        
        # Audio processing parameters
        self.target_sample_rate = 16000
        self.chunk_length_s = 30.0
        self.batch_size_alignment = 8
        
        print(f"🎯 Initializing WhisperX Engine (Fixed)")
        print(f"   Model: {model_size}")
        print(f"   Device: {self.device}")
        print(f"   Compute Type: {compute_type}")
        print(f"   Built-in Diarization: {'✅' if enable_diarization else '❌'}")
        print(f"   Batch Size: {batch_size}")
        
        self._load_whisperx_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = "cpu"
                print(f"   ⚠️  Using CPU (CUDA not available)")
        return device
    
    def _load_whisperx_models(self):
        """Load WhisperX models with correct API usage"""
        try:
            import whisperx
            
            print(f"📥 Loading WhisperX models...")
            
            # FIXED: Load main transcription model with correct parameters only
            print(f"   Loading transcription model ({self.model_size})...")
            self.whisperx_model = whisperx.load_model(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
                # REMOVED: vad_filter and vad_parameters - these are not valid arguments
            )
            print(f"      ✅ Transcription model loaded")
            
            # Pre-load alignment models for major languages
            print(f"   Pre-loading alignment models...")
            major_languages = ["en", "de", "fr", "es", "it"]
            
            for lang in major_languages:
                try:
                    model, metadata = whisperx.load_align_model(
                        language_code=lang, 
                        device=self.device
                    )
                    self.alignment_models[lang] = (model, metadata)
                    print(f"      ✅ Alignment model loaded: {lang}")
                except Exception as e:
                    print(f"      ⚠️  Alignment model failed for {lang}: {e}")
            
            # Load diarization pipeline if enabled
            if self.enable_diarization:
                print(f"   Loading speaker diarization pipeline...")
                try:
                    self.diarization_pipeline = whisperx.DiarizationPipeline(
                        use_auth_token=True,
                        device=self.device
                    )
                    print(f"      ✅ Diarization pipeline loaded")
                except Exception as e:
                    print(f"      ⚠️  Diarization pipeline failed: {e}")
                    print(f"      💡 Make sure you have HuggingFace token configured")
                    self.enable_diarization = False
            
            print(f"✅ WhisperX ready with enhanced capabilities")
            
        except ImportError:
            raise ImportError(
                "WhisperX not installed. Install with: pip install whisperx"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load WhisperX: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded WhisperX model"""
        return {
            "model_name": "whisperx",
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "target_sample_rate": self.target_sample_rate,
            "diarization_enabled": self.enable_diarization,
            "alignment_languages": list(self.alignment_models.keys()),
            "batch_size": self.batch_size
        }
    
    def _load_and_preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio for WhisperX"""
        try:
            # Load audio with librosa
            audio_data, original_sr = librosa.load(
                str(audio_path),
                sr=self.target_sample_rate,
                mono=True,
                res_type='kaiser_best'
            )
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        enable_diarization: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict:
        """
        Transcribe audio using WhisperX with forced alignment and optional diarization
        
        Args:
            audio_path: Path to audio file
            language: Language code ('en', 'de', etc.) or None for auto-detection
            task: Task type ('transcribe' or 'translate')
            word_timestamps: Enable precise word-level timestamps via forced alignment
            enable_diarization: Override diarization setting for this transcription
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            
        Returns:
            Dictionary with enhanced transcription results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Use instance diarization setting if not overridden
        if enable_diarization is None:
            enable_diarization = self.enable_diarization
        
        print(f"🎯 WhisperX Transcription: {audio_path.name}")
        print(f"   Model: {self.model_size}")
        print(f"   Language: {language or 'Auto-detect'}")
        print(f"   Word Alignment: {'✅' if word_timestamps else '❌'}")
        print(f"   Speaker Diarization: {'✅' if enable_diarization else '❌'}")
        
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            print("📥 Loading audio with optimal settings...")
            audio_data = self._load_and_preprocess_audio(audio_path)
            audio_duration = len(audio_data) / self.target_sample_rate
            
            print(f"📊 Audio duration: {audio_duration:.1f}s")
            
            # Step 1: Initial transcription with WhisperX
            print("🔄 Running WhisperX transcription...")
            transcription_start = time.time()
            
            # FIXED: Use correct transcribe API
            result = self.whisperx_model.transcribe(
                audio_data,
                batch_size=self.batch_size,
                language=language,
                task=task
            )
            
            transcription_time = time.time() - transcription_start
            detected_language = result.get("language", language or "unknown")
            
            print(f"   ✅ Transcription complete ({transcription_time:.1f}s)")
            print(f"   🗣️  Language detected: {detected_language}")
            print(f"   📝 Segments: {len(result.get('segments', []))}")
            
            # Step 2: Forced alignment for precise word timestamps
            alignment_time = 0
            if word_timestamps:
                print("🔄 Applying forced alignment for precise timestamps...")
                alignment_start = time.time()
                
                result = self._apply_forced_alignment(result, audio_data, detected_language)
                
                alignment_time = time.time() - alignment_start
                print(f"   ✅ Forced alignment complete ({alignment_time:.1f}s)")
            
            # Step 3: Speaker diarization (if enabled)
            diarization_time = 0
            if enable_diarization and self.diarization_pipeline:
                print("🔄 Running speaker diarization...")
                diarization_start = time.time()
                
                result = self._apply_speaker_diarization(
                    result, audio_data, min_speakers, max_speakers
                )
                
                diarization_time = time.time() - diarization_start
                print(f"   ✅ Speaker diarization complete ({diarization_time:.1f}s)")
            
            # Calculate processing statistics
            total_processing_time = time.time() - start_time
            speed_ratio = audio_duration / total_processing_time if total_processing_time > 0 else 0
            
            # Extract full text
            full_text = " ".join([seg.get("text", "").strip() for seg in result.get("segments", [])])
            
            print(f"✅ WhisperX processing complete!")
            print(f"📝 Text length: {len(full_text)} characters")
            print(f"⏱️  Processing time: {total_processing_time:.1f}s")
            print(f"⚡ Speed ratio: {speed_ratio:.1f}x real-time")
            
            # Create enhanced result
            enhanced_result = {
                "text": full_text,
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", []) if word_timestamps else [],
                "metadata": {
                    "file_name": audio_path.name,
                    "file_size_mb": audio_path.stat().st_size / 1e6,
                    "audio_duration_seconds": audio_duration,
                    "processing_time_seconds": total_processing_time,
                    "speed_ratio": speed_ratio,
                    "model_name": "whisperx",
                    "model_size": self.model_size,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "word_timestamps": word_timestamps,
                    "task": task,
                    "engine": "whisperx",
                    "forced_alignment_applied": word_timestamps,
                    "diarization_applied": enable_diarization and self.diarization_pipeline is not None,
                    "batch_size": self.batch_size,
                    "transcription_time": transcription_time,
                    "alignment_time": alignment_time,
                    "diarization_time": diarization_time
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ WhisperX transcription failed: {e}")
            raise RuntimeError(f"WhisperX transcription failed: {e}")
    
    def _apply_forced_alignment(self, result: Dict, audio_data: np.ndarray, language: str) -> Dict:
        """Apply forced alignment for precise word-level timestamps"""
        try:
            import whisperx
            
            # Get alignment model for detected language
            if language in self.alignment_models:
                align_model, align_metadata = self.alignment_models[language]
                print(f"      Using cached alignment model: {language}")
            else:
                print(f"      Loading alignment model for: {language}")
                try:
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=language, 
                        device=self.device
                    )
                    # Cache for future use
                    self.alignment_models[language] = (align_model, align_metadata)
                except Exception as e:
                    print(f"      ⚠️  Alignment model unavailable for {language}, using English")
                    if "en" in self.alignment_models:
                        align_model, align_metadata = self.alignment_models["en"]
                    else:
                        align_model, align_metadata = whisperx.load_align_model(
                            language_code="en", 
                            device=self.device
                        )
                        self.alignment_models["en"] = (align_model, align_metadata)
            
            # Apply forced alignment
            aligned_result = whisperx.align(
                result["segments"], 
                align_model, 
                align_metadata, 
                audio_data, 
                self.device, 
                return_char_alignments=False,  # Set to False for stability
                batch_size=self.batch_size_alignment
            )
            
            # Merge aligned results back
            result["segments"] = aligned_result["segments"]
            result["word_segments"] = aligned_result.get("word_segments", [])
            
            # Count words with precise timestamps
            word_count = sum(len(seg.get("words", [])) for seg in result["segments"])
            print(f"      📝 Precise timestamps: {word_count} words")
            
            return result
            
        except Exception as e:
            print(f"      ⚠️  Forced alignment failed: {e}")
            return result
    
    def _apply_speaker_diarization(
        self, 
        result: Dict, 
        audio_data: np.ndarray, 
        min_speakers: Optional[int], 
        max_speakers: Optional[int]
    ) -> Dict:
        """Apply speaker diarization using WhisperX built-in capabilities"""
        try:
            import whisperx
            
            if not self.diarization_pipeline:
                print(f"      ⚠️  Diarization pipeline not available")
                return result
            
            # Create temporary audio file for diarization
            temp_audio_path = None
            try:
                # Save audio to temporary file
                temp_fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
                os.close(temp_fd)
                sf.write(temp_audio_path, audio_data, self.target_sample_rate)
                
                # Run diarization
                diarization_segments = self.diarization_pipeline(
                    temp_audio_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                
                # Assign speakers to segments
                result = whisperx.assign_word_speakers(
                    diarization_segments, 
                    result
                )
                
                # Count speakers found
                speakers = set()
                for segment in result["segments"]:
                    if "speaker" in segment:
                        speakers.add(segment["speaker"])
                
                print(f"      👥 Speakers identified: {len(speakers)}")
                if speakers:
                    print(f"      🎙️  Speaker labels: {', '.join(sorted(speakers))}")
                
                return result
                
            finally:
                # Clean up temporary file
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    
        except Exception as e:
            print(f"      ⚠️  Speaker diarization failed: {e}")
            return result
    
    def get_word_level_timestamps(self, results: Dict) -> List[Dict]:
        """Extract precise word-level timestamps from WhisperX results"""
        words = []
        
        for segment in results.get('segments', []):
            if 'words' in segment:
                segment_speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                
                for word_info in segment['words']:
                    word_dict = {
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0),
                        'confidence': word_info.get('probability', word_info.get('score', 1.0)),
                        'speaker': word_info.get('speaker', segment_speaker)
                    }
                    words.append(word_dict)
        
        return words
    
    def extract_speaker_segments(self, results: Dict) -> List[Dict]:
        """Extract speaker-labeled segments from WhisperX results"""
        segments = []
        
        for segment in results.get('segments', []):
            seg_dict = {
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0),
                'text': segment.get('text', '').strip(),
                'speaker': segment.get('speaker', 'SPEAKER_UNKNOWN'),
                'words': segment.get('words', [])
            }
            segments.append(seg_dict)
        
        return segments
    
    def get_speakers_list(self, results: Dict) -> List[str]:
        """Extract unique speakers from WhisperX results"""
        speakers = set()
        
        for segment in results.get('segments', []):
            if 'speaker' in segment:
                speakers.add(segment['speaker'])
        
        return sorted(list(speakers))


# Test function
def test_whisperx_installation():
    """Test if WhisperX is properly installed"""
    try:
        import whisperx
        print("✅ WhisperX is properly installed")
        return True
    except ImportError:
        print("❌ WhisperX not installed")
        print("   Install with: pip install whisperx")
        return False


if __name__ == "__main__":
    print("🔧 Testing Fixed WhisperX Engine")
    
    if test_whisperx_installation():
        try:
            engine = WhisperXEngine(
                model_size="base",  # Use smaller model for testing
                device="auto",
                compute_type="float16",
                enable_diarization=True,
                batch_size=8
            )
            print("✅ WhisperX Engine initialized successfully!")
            
        except Exception as e:
            print(f"❌ WhisperX Engine initialization failed: {e}")
    else:
        print("Please install WhisperX first: pip install whisperx")