# pyannote_engine.py - PyAnnote Speaker Diarization Engine (Fixed)

from pyannote.audio import Pipeline
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import librosa
import soundfile as sf
import tempfile
import os
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class PyAnnoteEngine:
    """
    PyAnnote Speaker Diarization Engine
    Identifies who is speaking when in audio files
    Fixed version with better audio handling
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
                print(f"🎯 Using GPU for diarization")
            else:
                device = "cpu"
                print("⚠️  Using CPU for diarization")
        
        return device
    
    def _load_pipeline(self):
        """Load PyAnnote diarization pipeline"""
        try:
            print(f"📥 Loading PyAnnote diarization pipeline...")
            
            with tqdm(total=100, desc="Loading diarization", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                pbar.update(20)
                
                # Load the pre-trained diarization pipeline
                if self.use_auth_token:
                    # This requires HuggingFace token
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=True
                    )
                else:
                    # Alternative model that might not require token
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                
                pbar.update(60)
                
                # Move to GPU if available
                if self.device == "cuda" and torch.cuda.is_available():
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                
                pbar.update(20)
                pbar.set_description("Diarization pipeline ready")
            
            print(f"✅ PyAnnote ready")
            
        except Exception as e:
            print(f"❌ Failed to load PyAnnote pipeline: {e}")
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure you have a HuggingFace account")
            print("2. Accept the model license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("3. Get your token at: https://huggingface.co/settings/tokens")
            print("4. Run: huggingface-cli login")
            raise
    
    def _preprocess_audio_for_pyannote(self, audio_path: Path) -> str:
        """
        Preprocess audio to ensure compatibility with PyAnnote
        
        Args:
            audio_path: Original audio file path
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            print("🔧 Preprocessing audio for PyAnnote...")
            
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                str(audio_path), 
                sr=16000,  # PyAnnote expects 16kHz
                mono=True,
                res_type='kaiser_fast'  # Faster resampling
            )
            
            # Ensure audio length is compatible
            # PyAnnote works better with certain length constraints
            min_duration = 1.0  # Minimum 1 second
            max_duration = 3600.0  # Maximum 1 hour
            
            duration = len(audio_data) / 16000
            
            if duration < min_duration:
                # Pad short audio
                target_length = int(min_duration * 16000)
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
                print(f"   📏 Padded audio from {duration:.1f}s to {min_duration:.1f}s")
            
            elif duration > max_duration:
                # Trim long audio
                target_length = int(max_duration * 16000)
                audio_data = audio_data[:target_length]
                print(f"   ✂️  Trimmed audio from {duration:.1f}s to {max_duration:.1f}s")
            
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
            
            final_duration = len(audio_data) / 16000
            print(f"   ✅ Audio preprocessed: {final_duration:.1f}s duration")
            
            return temp_path
            
        except Exception as e:
            print(f"❌ Audio preprocessing failed: {e}")
            raise
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
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
        
        print(f"🎯 Diarizing: {audio_path.name}")
        
        processed_audio_path = None
        
        try:
            # Preprocess audio to ensure compatibility
            processed_audio_path = self._preprocess_audio_for_pyannote(audio_path)
            
            # Perform diarization with progress bar
            with tqdm(total=100, desc="Speaker diarization", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed} | ETA: {remaining}') as pbar:
                
                pbar.update(10)
                
                # Apply diarization pipeline using preprocessed audio
                try:
                    if num_speakers:
                        # Fixed number of speakers
                        diarization = self.pipeline(
                            processed_audio_path,
                            num_speakers=num_speakers
                        )
                    else:
                        # Auto-detect number of speakers
                        diarization = self.pipeline(
                            processed_audio_path,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                except RuntimeError as e:
                    if "tensor" in str(e).lower() and "size" in str(e).lower():
                        print(f"\n⚠️  Tensor size error detected. Trying fallback approach...")
                        
                        # Fallback: Use smaller chunks or different parameters
                        if not num_speakers:
                            # Try with fixed number of speakers as fallback
                            print(f"   🔄 Trying with fixed 2 speakers...")
                            diarization = self.pipeline(
                                processed_audio_path,
                                num_speakers=2
                            )
                        else:
                            # Re-raise the original error if we can't fallback
                            raise
                    else:
                        raise
                
                pbar.update(80)
                
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
                
                pbar.update(10)
                pbar.set_description("Diarization complete")
            
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
            
            print(f"✅ Diarization complete!")
            print(f"   Speakers detected: {len(speakers)}")
            print(f"   Total segments: {len(segments)}")
            print(f"   Audio duration: {total_duration:.1f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Diarization failed: {e}")
            
            # Provide specific troubleshooting based on error type
            error_str = str(e).lower()
            if "tensor" in error_str and "size" in error_str:
                print(f"\n💡 This appears to be a tensor size mismatch error.")
                print(f"   Possible solutions:")
                print(f"   1. Try with a different audio file")
                print(f"   2. Convert audio to WAV format first")
                print(f"   3. Try with num_speakers=2 (fixed number)")
                print(f"   4. Check if audio file is corrupted")
            elif "memory" in error_str:
                print(f"\n💡 This appears to be a memory error.")
                print(f"   Try using a smaller model or shorter audio file.")
            
            raise
        finally:
            # Always clean up temporary files
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
    
    def print_speaker_stats(self, results: Dict):
        """Print detailed speaker statistics"""
        print(f"\n📊 Speaker Statistics:")
        print(f"   Total duration: {results['total_duration']:.1f}s")
        print(f"   Number of speakers: {results['num_speakers']}")
        print(f"   Total segments: {len(results['segments'])}")
        
        print(f"\n👥 Per-speaker breakdown:")
        for speaker, stats in results['speaker_stats'].items():
            print(f"   {speaker}:")
            print(f"      Segments: {stats['segments']}")
            print(f"      Duration: {stats['total_duration']:.1f}s ({stats['percentage']:.1f}%)")
    
    def save_diarization_results(self, results: Dict, output_path: Union[str, Path]):
        """
        Save diarization results in RTTM format (standard for speaker diarization)
        
        Args:
            results: Results from diarize_audio()
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for segment in results['segments']:
                # RTTM format: SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                f.write(f"SPEAKER {results['metadata']['file_name']} 1 {segment['start']:.3f} {segment['duration']:.3f} <NA> <NA> {segment['speaker']} <NA> <NA>\n")
        
        print(f"💾 Saved diarization: {output_path.name}")


def test_pyannote_setup():
    """Test if PyAnnote is properly configured"""
    print("🔍 Testing PyAnnote Setup...")
    
    try:
        # Test HuggingFace authentication
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ HuggingFace authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ HuggingFace authentication failed: {e}")
        print("\n💡 To fix this:")
        print("1. Create account at: https://huggingface.co/join")
        print("2. Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("3. Get token at: https://huggingface.co/settings/tokens")
        print("4. Run: huggingface-cli login")
        return False
    
    try:
        # Test PyAnnote import
        from pyannote.audio import Pipeline
        print("✅ PyAnnote audio imported successfully")
    except ImportError as e:
        print(f"❌ PyAnnote import failed: {e}")
        print("Run: pip install pyannote.audio")
        return False
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    print("✅ PyAnnote setup looks good!")
    return True


def main():
    """Test PyAnnote engine with improved error handling"""
    
    # Test setup first
    if not test_pyannote_setup():
        print("❌ Setup incomplete. Please fix the issues above.")
        return
    
    # Test with sample audio file
    from pathlib import Path
    
    # Look for audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.mp4', '*.m4a']:
        audio_files.extend(Path('.').glob(ext))
    
    if not audio_files:
        print("❌ No audio files found for testing")
        print("   Place an audio file in the current directory")
        return
    
    audio_file = audio_files[0]
    print(f"🎯 Testing with: {audio_file.name}")
    
    try:
        # Initialize engine
        engine = PyAnnoteEngine()
        
        # Perform diarization with error handling
        print(f"\n🔄 Testing diarization...")
        results = engine.diarize_audio(
            audio_file,
            num_speakers=None,  # Auto-detect
            min_speakers=1,
            max_speakers=5  # Reduce max for testing
        )
        
        # Show results
        engine.print_speaker_stats(results)
        
        # Save RTTM file
        engine.save_diarization_results(results, f"output/{audio_file.stem}_diarization.rttm")
        
        print(f"\n🎉 PyAnnote test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"\n💡 Try running the complete pipeline with: python pipeline_integration.py")


if __name__ == "__main__":
    main()