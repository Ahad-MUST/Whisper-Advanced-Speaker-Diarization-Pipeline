# whisper_engine.py - Local Whisper Speech-to-Text Engine (Progress Bar + Excel Output)

import whisper
import torch
import json
import time
import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import glob
import librosa
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Triton.*")

# Set up clean logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperEngine:
    """
    Local Whisper Engine for Speech-to-Text conversion
    Supports GPU acceleration and multiple model sizes
    FFmpeg-free implementation with progress bar
    """
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
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
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🚀 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = "cpu"
                print("⚠️  Using CPU (CUDA not available)")
        
        return device
    
    def _load_model(self):
        """Load Whisper model with error handling"""
        try:
            print(f"📥 Loading Whisper {self.model_size} model...")
            
            # Show progress bar for model loading
            with tqdm(total=100, desc="Loading model", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                start_time = time.time()
                
                # Simulate progress during model loading
                pbar.update(30)
                
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device
                )
                
                pbar.update(70)
                load_time = time.time() - start_time
                pbar.set_description(f"Model loaded ({load_time:.1f}s)")
                pbar.update(100)
            
            print(f"✅ Model ready")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _load_audio_with_librosa(self, audio_path: Path) -> np.ndarray:
        """
        Load and preprocess audio using librosa (no FFmpeg needed)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio data as numpy array (16kHz, mono)
        """
        try:
            print("🎵 Loading audio...")
            
            # Load audio with librosa (handles multiple formats without FFmpeg)
            with tqdm(total=100, desc="Loading audio", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed}') as pbar:
                pbar.update(20)
                
                audio_data, sample_rate = librosa.load(
                    str(audio_path), 
                    sr=16000,  # Whisper expects 16kHz
                    mono=True   # Convert to mono
                )
                pbar.update(60)
                
                # Ensure audio is float32 and in correct range
                audio_data = audio_data.astype(np.float32)
                
                # Whisper expects audio in range [-1, 1]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                pbar.update(20)
                duration = len(audio_data) / 16000
                pbar.set_description(f"Audio loaded ({duration:.1f}s)")
            
            print(f"✅ Audio ready: {duration:.1f}s duration")
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Audio loading failed: {e}")
            raise
    
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
            audio_path: Path to audio file (MP3, WAV, etc.)
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
        
        file_size_mb = audio_path.stat().st_size / 1e6
        print(f"📄 File: {audio_path.name} ({file_size_mb:.1f} MB)")
        
        start_time = time.time()
        
        try:
            # Load audio with librosa (bypasses FFmpeg completely)
            audio_data = self._load_audio_with_librosa(audio_path)
            audio_duration = len(audio_data) / 16000
            
            # Transcribe with Whisper using audio data directly
            print("🔄 Transcribing...")
            
            # Custom progress bar for transcription
            with tqdm(total=100, desc="Transcribing", bar_format='{desc}: {percentage:3.0f}%|{bar}| {elapsed} | ETA: {remaining}') as pbar:
                
                # Suppress Whisper's verbose output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Start transcription
                    pbar.update(10)
                    
                    result = self.model.transcribe(
                        audio_data,  # Pass numpy array directly instead of file path
                        language=language,
                        task=task,
                        word_timestamps=word_timestamps,
                        verbose=False  # Disable verbose output for clean logs
                    )
                    
                    pbar.update(90)
                    pbar.set_description("Transcription complete")
            
            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            # Show completion message
            print(f"✅ Transcription complete!")
            print(f"   Language: {result['language']}")
            print(f"   Duration: {audio_duration:.1f}s")
            print(f"   Processing: {processing_time:.1f}s ({speed_ratio:.1f}x real-time)")
            
            # Enhance result with metadata
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
            print(f"❌ Transcription failed: {e}")
            raise
    
    def save_results(self, results: Dict, output_path: Union[str, Path], format: str = "json"):
        """
        Save transcription results to file
        
        Args:
            results: Transcription results from transcribe_audio()
            output_path: Output file path
            format: Output format ('json', 'txt', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcription: {results['metadata']['file_name']}\n")
                f.write(f"Language: {results['language']}\n")
                f.write(f"Duration: {results['metadata']['audio_duration_seconds']:.2f}s\n")
                f.write(f"Processing Time: {results['metadata']['processing_time_seconds']:.2f}s\n")
                f.write(f"Speed Ratio: {results['metadata']['speed_ratio']:.2f}x real-time\n")
                f.write("-" * 60 + "\n\n")
                f.write(results['text'])
                
        elif format.lower() == "excel":
            # Create DataFrame for segments
            segments_data = []
            for i, segment in enumerate(results['segments'], 1):
                segments_data.append({
                    'Segment': i,
                    'Start_Time': segment['start'],
                    'End_Time': segment['end'],
                    'Duration': segment['end'] - segment['start'],
                    'Text': segment['text'].strip()
                })
            
            df_segments = pd.DataFrame(segments_data)
            
            # Create DataFrame for metadata
            metadata_data = [
                ['File Name', results['metadata']['file_name']],
                ['Language', results['language']],
                ['Audio Duration (s)', results['metadata']['audio_duration_seconds']],
                ['Processing Time (s)', results['metadata']['processing_time_seconds']],
                ['Speed Ratio', f"{results['metadata']['speed_ratio']:.2f}x"],
                ['Model Size', results['metadata']['model_size']],
                ['Device', results['metadata']['device']],
                ['Total Segments', len(results['segments'])],
                ['Total Characters', len(results['text'])],
                ['']  # Empty row for separation
            ]
            
            df_metadata = pd.DataFrame(metadata_data, columns=['Property', 'Value'])
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_metadata.to_excel(writer, sheet_name='Summary', index=False)
                df_segments.to_excel(writer, sheet_name='Segments', index=False)
                
                # Add full text as a separate sheet
                df_fulltext = pd.DataFrame([{'Full_Transcript': results['text']}])
                df_fulltext.to_excel(writer, sheet_name='Full_Text', index=False)
        
        # Only print filename, not full path
        print(f"💾 Saved: {output_path.name}")
    
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


def find_audio_files(directory: str = ".") -> List[Path]:
    """Find all audio files in the specified directory"""
    audio_extensions = ["*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    audio_files = []
    
    for extension in audio_extensions:
        # Case insensitive search
        files = list(Path(directory).glob(extension))
        files.extend(list(Path(directory).glob(extension.upper())))
        audio_files.extend(files)
    
    # Remove duplicates and sort
    unique_files = []
    seen = set()
    for file in audio_files:
        if file.resolve() not in seen:
            seen.add(file.resolve())
            unique_files.append(file)
    
    return sorted(unique_files)


def select_audio_file() -> Optional[Path]:
    """Let user select an audio file from available files"""
    audio_files = find_audio_files()
    
    if not audio_files:
        print("❌ No audio files found!")
        print("   Supported: MP3, WAV, MP4, M4A, FLAC, OGG, AAC")
        return None
    
    if len(audio_files) == 1:
        print(f"📁 Found: {audio_files[0].name}")
        return audio_files[0]
    
    print("📁 Audio files found:")
    for i, file in enumerate(audio_files, 1):
        size_mb = file.stat().st_size / 1e6
        print(f"   {i}. {file.name} ({size_mb:.1f} MB)")
    
    while True:
        try:
            choice = input(f"\nSelect file (1-{len(audio_files)}) or Enter for first: ").strip()
            
            if not choice:  # Default to first file
                return audio_files[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(audio_files):
                return audio_files[choice_idx]
            else:
                print(f"Please enter 1-{len(audio_files)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return None


def main():
    """Example usage and testing"""
    
    # Configuration
    MODEL_SIZE = "large"  # Options: tiny, base, small, medium, large
    OUTPUT_DIR = "output"
    
    try:
        print("🎤 Whisper Speech-to-Text Engine")
        print("=" * 40)
        
        # Find and select audio file
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        # Initialize Whisper Engine
        print(f"\n🔧 Initializing...")
        whisper_engine = WhisperEngine(model_size=MODEL_SIZE)
        
        # Transcribe audio
        print(f"\n🎯 Processing...")
        results = whisper_engine.transcribe_audio(
            audio_path=audio_file,
            language=None,  # Auto-detect language
            word_timestamps=True
        )
        
        # Save results in TXT, JSON, and Excel formats
        print(f"\n💾 Saving results...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = audio_file.stem
        
        # Show progress bar for saving
        with tqdm(total=3, desc="Saving files", bar_format='{desc}: {n}/{total} files') as pbar:
            whisper_engine.save_results(results, f"{OUTPUT_DIR}/{base_name}.json", "json")
            pbar.update(1)
            
            whisper_engine.save_results(results, f"{OUTPUT_DIR}/{base_name}.txt", "txt")
            pbar.update(1)
            
            whisper_engine.save_results(results, f"{OUTPUT_DIR}/{base_name}.xlsx", "excel")
            pbar.update(1)
        
        # Summary
        words = whisper_engine.get_word_level_timestamps(results)
        print(f"\n✅ Summary:")
        print(f"   📄 Segments: {len(results['segments'])}")
        print(f"   📝 Words: {len(words)}")
        print(f"   📝 Characters: {len(results['text'])}")
        print(f"   ⏱️  Duration: {results['metadata']['audio_duration_seconds']:.1f}s")
        print(f"   🚀 Speed: {results['metadata']['speed_ratio']:.1f}x real-time")
        print(f"   🗂️  Language: {results['language']}")
        print(f"   📁 Files saved in '{OUTPUT_DIR}/' directory")
        
        print(f"\n🎉 Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()