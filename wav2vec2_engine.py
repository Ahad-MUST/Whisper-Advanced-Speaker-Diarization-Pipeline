# wav2vec2_engine.py - Wav2Vec2 ASR Engine for German Language

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

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class Wav2Vec2Engine:
    """
    Wav2Vec2 ASR Engine optimized for German language
    
    Uses Facebook's Wav2Vec2 models with German language specialization:
    - wav2vec2-large-xlsr-53-german for German ASR
    - wav2vec2-base-960h for English fallback
    - Optimized for medical/consultation audio
    """
    
    def __init__(self, model_name: str = "german", device: str = "auto"):
        """
        Initialize Wav2Vec2 Engine
        
        Args:
            model_name: Model to use ('german', 'english', 'multilingual')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.temp_files = []
        
        # Model configurations
        self.model_configs = {
            'german': {
                'model_id': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
                'language': 'de',
                'description': 'German-optimized Wav2Vec2 model'
            },
            'english': {
                'model_id': 'facebook/wav2vec2-base-960h',
                'language': 'en', 
                'description': 'English Wav2Vec2 model'
            },
            'multilingual': {
                'model_id': 'facebook/wav2vec2-large-xlsr-53',
                'language': 'auto',
                'description': 'Multilingual Wav2Vec2 model'
            }
        }
        
        # Audio processing parameters
        self.target_sample_rate = 16000
        self.chunk_length_s = 30.0  # Process in 30-second chunks
        self.overlap_s = 2.0        # 2-second overlap between chunks
        
        print(f"🎯 Initializing Wav2Vec2 Engine")
        print(f"   Model: {self.model_configs[model_name]['description']}")
        print(f"   Language: {self.model_configs[model_name]['language']}")
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"   Device: GPU (CUDA)")
            else:
                device = "cpu"
                print(f"   Device: CPU")
        return device
    
    def _load_model(self):
        """Load Wav2Vec2 model and processor"""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            model_id = self.model_configs[self.model_name]['model_id']
            
            print(f"📥 Loading {self.model_name} Wav2Vec2 model...")
            
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Wav2Vec2 model loaded successfully")
            
        except ImportError:
            raise ImportError("Transformers library not installed. Run: pip install transformers")
        except Exception as e:
            print(f"❌ Failed to load Wav2Vec2 model: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_id": self.model_configs[self.model_name]['model_id'],
            "language": self.model_configs[self.model_name]['language'],
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "target_sample_rate": self.target_sample_rate
        }
    
    def _load_and_preprocess_audio(self, audio_path: Path) -> np.ndarray:
        """Load and preprocess audio for Wav2Vec2"""
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
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def _create_chunks_with_timestamps(self, audio_data: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Create overlapping chunks with timestamps"""
        chunks = []
        
        chunk_samples = int(self.chunk_length_s * self.target_sample_rate)
        overlap_samples = int(self.overlap_s * self.target_sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        for start_sample in range(0, len(audio_data), step_samples):
            end_sample = min(start_sample + chunk_samples, len(audio_data))
            
            chunk = audio_data[start_sample:end_sample]
            start_time = start_sample / self.target_sample_rate
            end_time = end_sample / self.target_sample_rate
            
            # Pad chunk if too short
            if len(chunk) < chunk_samples and start_sample == 0:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            chunks.append((chunk, start_time, end_time))
        
        return chunks
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk"""
        try:
            # Process audio with Wav2Vec2 processor
            inputs = self.processor(
                audio_chunk, 
                sampling_rate=self.target_sample_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"⚠️  Chunk transcription failed: {e}")
            return ""
    
    def _create_word_timestamps(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """Create approximate word timestamps (Wav2Vec2 doesn't provide word-level timestamps by default)"""
        words = text.split()
        if not words:
            return []
        
        duration = end_time - start_time
        word_duration = duration / len(words)
        
        word_timestamps = []
        for i, word in enumerate(words):
            word_start = start_time + (i * word_duration)
            word_end = word_start + word_duration
            
            word_timestamps.append({
                'word': word,
                'start': word_start,
                'end': word_end,
                'confidence': 1.0  # Wav2Vec2 doesn't provide word-level confidence
            })
        
        return word_timestamps
    
    def _stitch_chunks(self, chunk_results: List[Tuple[str, float, float]]) -> Tuple[str, List[Dict]]:
        """Stitch transcribed chunks together with deduplication"""
        if not chunk_results:
            return "", []
        
        full_text = ""
        all_segments = []
        
        for i, (text, start_time, end_time) in enumerate(chunk_results):
            if not text.strip():
                continue
            
            # Handle overlap deduplication for chunks after the first
            if i > 0 and self.overlap_s > 0:
                # Simple deduplication: remove overlapping words
                words = text.split()
                overlap_words = int(len(words) * (self.overlap_s / self.chunk_length_s))
                if overlap_words > 0 and len(words) > overlap_words:
                    text = " ".join(words[overlap_words:])
                    # Adjust start time
                    start_time += self.overlap_s
            
            if text.strip():
                # Create segment
                segment = {
                    'start': start_time,
                    'end': end_time,
                    'text': text.strip(),
                    'words': self._create_word_timestamps(text.strip(), start_time, end_time)
                }
                all_segments.append(segment)
                
                # Add to full text
                if full_text:
                    full_text += " " + text.strip()
                else:
                    full_text = text.strip()
        
        return full_text, all_segments
    
    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file using Wav2Vec2
        
        Args:
            audio_path: Path to audio file
            language: Language hint (used for model selection)
            task: Task type ('transcribe' only for Wav2Vec2)
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🎯 Transcribing with Wav2Vec2: {audio_path.name}")
        start_time = time.time()
        
        try:
            # Load and preprocess audio
            print("📥 Loading audio...")
            audio_data = self._load_and_preprocess_audio(audio_path)
            audio_duration = len(audio_data) / self.target_sample_rate
            
            print(f"📊 Audio duration: {audio_duration:.1f}s")
            print(f"📊 Sample rate: {self.target_sample_rate}Hz")
            
            # Create chunks
            print("🔄 Creating audio chunks...")
            chunks = self._create_chunks_with_timestamps(audio_data)
            print(f"📦 Created {len(chunks)} chunks")
            
            # Transcribe chunks
            print("🔄 Transcribing chunks...")
            chunk_results = []
            
            for i, (chunk, chunk_start, chunk_end) in enumerate(chunks):
                print(f"   Processing chunk {i+1}/{len(chunks)} ({chunk_start:.1f}s-{chunk_end:.1f}s)")
                
                transcription = self._transcribe_chunk(chunk)
                if transcription:
                    chunk_results.append((transcription, chunk_start, chunk_end))
            
            # Stitch results together
            print("🔗 Stitching chunks together...")
            full_text, segments = self._stitch_chunks(chunk_results)
            
            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time if processing_time > 0 else 0
            
            # Detect language based on model
            detected_language = self.model_configs[self.model_name]['language']
            if detected_language == 'auto':
                # Simple language detection based on content (very basic)
                detected_language = self._detect_language_simple(full_text)
            
            print(f"✅ Wav2Vec2 transcription complete!")
            print(f"📝 Text length: {len(full_text)} characters")
            print(f"🗣️  Language: {detected_language}")
            print(f"⏱️  Processing time: {processing_time:.1f}s")
            print(f"⚡ Speed ratio: {speed_ratio:.1f}x real-time")
            
            # Create enhanced result
            enhanced_result = {
                "text": full_text,
                "language": detected_language,
                "segments": segments,
                "metadata": {
                    "file_name": audio_path.name,
                    "file_size_mb": audio_path.stat().st_size / 1e6,
                    "audio_duration_seconds": audio_duration,
                    "processing_time_seconds": processing_time,
                    "speed_ratio": speed_ratio,
                    "model_name": self.model_name,
                    "model_id": self.model_configs[self.model_name]['model_id'],
                    "device": self.device,
                    "word_timestamps": word_timestamps,
                    "task": task,
                    "chunks_processed": len(chunks),
                    "engine": "wav2vec2"
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ Wav2Vec2 transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            self._cleanup_temp_files()
    
    def _detect_language_simple(self, text: str) -> str:
        """Simple language detection based on common words"""
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # German indicators
        german_words = ["der", "die", "das", "und", "ist", "ich", "sie", "er", "ein", "eine", "mit", "zu", "auf", "für", "von", "werden", "haben", "sein"]
        german_count = sum(1 for word in german_words if word in text_lower)
        
        # English indicators  
        english_words = ["the", "and", "is", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this"]
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if german_count > english_count:
            return "de"
        elif english_count > german_count:
            return "en"
        else:
            return "unknown"
    
    def get_word_level_timestamps(self, results: Dict) -> List[Dict]:
        """
        Extract word-level timestamps from results
        
        Returns:
            List of dictionaries with word, start, end, confidence
        """
        words = []
        
        for segment in results.get('segments', []):
            if 'words' in segment:
                words.extend(segment['words'])
        
        return words
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self._cleanup_temp_files()


# Utility functions
def get_available_wav2vec2_models() -> Dict[str, Dict]:
    """Get list of available Wav2Vec2 models"""
    return {
        'german': {
            'model_id': 'jonatasgrosman/wav2vec2-large-xlsr-53-german',
            'language': 'de',
            'description': 'German-optimized Wav2Vec2 (RECOMMENDED for German)',
            'use_case': 'German medical consultations, interviews, meetings'
        },
        'english': {
            'model_id': 'facebook/wav2vec2-base-960h',
            'language': 'en',
            'description': 'English Wav2Vec2 model',
            'use_case': 'English conversations, meetings'
        },
        'multilingual': {
            'model_id': 'facebook/wav2vec2-large-xlsr-53',
            'language': 'auto',
            'description': 'Multilingual Wav2Vec2 model',
            'use_case': 'Mixed-language content, language detection'
        }
    }


def main():
    """Example usage of Wav2Vec2 engine"""
    
    print("🎯 Wav2Vec2 ASR Engine - German Language Specialist")
    print("=" * 60)
    
    # Show available models
    models = get_available_wav2vec2_models()
    print("📋 Available models:")
    for i, (key, info) in enumerate(models.items(), 1):
        print(f"   {i}. {key}: {info['description']}")
    
    # Find audio files
    audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio file(s):")
    for i, file in enumerate(audio_files, 1):
        size_mb = file.stat().st_size / 1e6
        print(f"   {i}. {file.name} ({size_mb:.1f} MB)")
    
    # Select file
    try:
        choice = input(f"\nSelect file (1-{len(audio_files)}) or Enter for first: ").strip()
        
        if not choice:
            selected_file = audio_files[0]
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(audio_files):
                selected_file = audio_files[choice_idx]
            else:
                print("Invalid selection")
                return
    except (ValueError, KeyboardInterrupt):
        print("Invalid selection or cancelled")
        return
    
    # Select model
    try:
        model_choice = input("\nSelect model (1-3) or Enter for German: ").strip()
        
        if not model_choice or model_choice == "1":
            model_name = "german"
        elif model_choice == "2":
            model_name = "english"
        elif model_choice == "3":
            model_name = "multilingual"
        else:
            model_name = "german"
            
    except KeyboardInterrupt:
        print("Cancelled")
        return
    
    print(f"\n🚀 Processing {selected_file.name} with {model_name} Wav2Vec2...")
    
    try:
        # Initialize engine
        engine = Wav2Vec2Engine(model_name=model_name, device="auto")
        
        # Transcribe audio
        results = engine.transcribe_audio(
            audio_path=selected_file,
            word_timestamps=True
        )
        
        print(f"\n📊 TRANSCRIPTION RESULTS:")
        print(f"📝 Text: {results['text'][:200]}...")
        print(f"🗣️  Language: {results['language']}")
        print(f"📦 Segments: {len(results['segments'])}")
        print(f"⏱️  Speed: {results['metadata']['speed_ratio']:.1f}x real-time")
        
        # Save results
        output_file = selected_file.parent / f"{selected_file.stem}_wav2vec2_{model_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Wav2Vec2 Transcription ({model_name})\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"File: {selected_file.name}\n")
            f.write(f"Model: {results['metadata']['model_id']}\n")
            f.write(f"Language: {results['language']}\n")
            f.write(f"Duration: {results['metadata']['audio_duration_seconds']:.1f}s\n")
            f.write(f"Processing Time: {results['metadata']['processing_time_seconds']:.1f}s\n\n")
            f.write("TRANSCRIPT:\n")
            f.write("-" * 30 + "\n")
            f.write(results['text'])
        
        print(f"\n💾 Transcript saved: {output_file}")
        print("🎉 Wav2Vec2 transcription complete!")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()