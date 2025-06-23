# basic_audio_preprocessor.py - Simple Audio Preprocessor for Speech Recognition

import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")

class BasicAudioPreprocessor:
    """
    Basic Audio Preprocessor for Speech Recognition Engines
    
    Performs only essential audio standardization:
    - Format conversion (mono, 16kHz)
    - Normalization (-1 to 1 range)
    - DC offset removal
    - Basic clipping prevention
    
    Optimized for:
    - Whisper
    - PyAnnote
    - SpeechBrain
    - NeMo
    - Wav2Vec2
    """
    
    def __init__(self):
        """Initialize basic audio preprocessor"""
        self.target_sr = 16000  # Standard sample rate for speech models
        self.target_channels = 1  # Mono audio
        self.temp_files = []
        
        print(f"🔧 Basic Audio Preprocessor initialized")
        print(f"   Target: 16kHz mono, normalized audio")
        print(f"   Purpose: Speech recognition engine compatibility")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_original: bool = False
    ) -> Tuple[str, Optional[str], Dict]:
        """
        Process audio with basic standardization
        
        Args:
            audio_path: Input audio file path
            output_path: Output file path (optional)
            save_original: Save original for comparison
            
        Returns:
            Tuple of (processed_audio_path, original_audio_path, processing_metrics)
        """
        
        audio_path = Path(audio_path)
        print(f"🔧 Basic audio processing: {audio_path.name}")
        start_time = time.time()
        
        # Load and analyze original audio
        original_audio, original_sr = self._load_audio(audio_path)
        original_duration = len(original_audio) / original_sr
        
        # Analyze original audio
        original_metrics = self._analyze_audio(original_audio, original_sr, "original")
        print(f"   📊 Original: {original_sr}Hz, {original_duration:.1f}s")
        print(f"   📉 Peak: {original_metrics['peak']:.3f}, RMS: {original_metrics['rms']:.3f}")
        
        # Apply basic processing
        processed_audio = self._apply_basic_processing(original_audio, original_sr)
        
        # Analyze processed audio
        processed_metrics = self._analyze_audio(processed_audio, self.target_sr, "processed")
        
        # Create processing report
        processing_report = self._create_processing_report(
            original_metrics, processed_metrics, original_duration
        )
        
        # Save processed audio
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_processed_16k.wav"
        
        sf.write(output_path, processed_audio, self.target_sr, subtype='PCM_16')
        
        # Save original for comparison if requested
        original_comparison_path = None
        if save_original:
            original_comparison_path = audio_path.parent / f"{audio_path.stem}_original.wav"
            # Save original as-is for comparison
            sf.write(original_comparison_path, original_audio, original_sr, subtype='PCM_16')
        
        processing_time = time.time() - start_time
        print(f"   ✅ Basic processing complete ({processing_time:.1f}s)")
        print(f"   📈 Changes: {processing_report['summary']}")
        print(f"   💾 Processed: {output_path}")
        if original_comparison_path:
            print(f"   💾 Original: {original_comparison_path}")
        
        # Cleanup temporary files
        self.cleanup()
        
        return str(output_path), str(original_comparison_path) if original_comparison_path else None, processing_report
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with robust handling"""
        try:
            # Use librosa for robust audio loading
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=None,  # Preserve original sample rate initially
                mono=False,
                res_type='kaiser_best'
            )
            
            # Handle channel configuration
            if audio_data.ndim > 1:
                if audio_data.shape[0] == 2:
                    # For stereo, intelligently choose better channel or average
                    left_rms = np.sqrt(np.mean(audio_data[0] ** 2))
                    right_rms = np.sqrt(np.mean(audio_data[1] ** 2))
                    
                    if left_rms > right_rms * 1.5:  # Left significantly louder
                        audio_data = audio_data[0]
                        print(f"   📻 Using left channel (stronger signal)")
                    elif right_rms > left_rms * 1.5:  # Right significantly louder
                        audio_data = audio_data[1]
                        print(f"   📻 Using right channel (stronger signal)")
                    else:
                        # Channels similar, average them
                        audio_data = np.mean(audio_data, axis=0)
                        print(f"   📻 Averaging stereo channels")
                else:
                    # Multi-channel: take first channel
                    audio_data = audio_data[0]
                    print(f"   📻 Using first channel from {audio_data.shape[0]}-channel audio")
            
            return audio_data.astype(np.float32), sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def _apply_basic_processing(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """Apply basic audio processing pipeline"""
        
        print(f"   🔄 Applying basic processing pipeline...")
        processed_audio = audio_data.copy()
        
        # Step 1: Remove DC offset
        processed_audio = self._remove_dc_offset(processed_audio)
        print(f"      ✅ DC offset removal")
        
        # Step 2: Resample to 16kHz if needed
        if original_sr != self.target_sr:
            processed_audio = self._resample_audio(processed_audio, original_sr)
            print(f"      ✅ Resampled: {original_sr}Hz → {self.target_sr}Hz")
        else:
            print(f"      ✅ Sample rate: already {self.target_sr}Hz")
        
        # Step 3: Normalize to [-1, 1] range
        processed_audio = self._normalize_audio(processed_audio)
        print(f"      ✅ Normalized to [-1, 1] range")
        
        # Step 4: Prevent clipping and ensure safe levels
        processed_audio = self._safe_limiting(processed_audio)
        print(f"      ✅ Safe limiting applied")
        
        return processed_audio
    
    def _remove_dc_offset(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio"""
        return audio_data - np.mean(audio_data)
    
    def _resample_audio(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            resampled = librosa.resample(
                audio_data,
                orig_sr=original_sr,
                target_sr=self.target_sr,
                res_type='kaiser_best'
            )
            return resampled.astype(np.float32)
        except Exception as e:
            print(f"      ⚠️  Resampling failed: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1e-10:  # Avoid division by zero
            # Normalize to [-1, 1] with slight headroom
            normalized = audio_data / max_abs * 0.95
            return normalized
        else:
            return audio_data
    
    def _safe_limiting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply safe limiting to prevent clipping"""
        # Ensure audio stays within [-1, 1] bounds
        limited = np.clip(audio_data, -1.0, 1.0)
        
        # Apply soft clipping for values very close to limits
        limited = np.tanh(limited * 0.95) * 0.95
        
        return limited
    
    def _analyze_audio(self, audio_data: np.ndarray, sample_rate: int, stage: str) -> Dict:
        """Analyze audio characteristics"""
        try:
            # Basic statistics
            rms = np.sqrt(np.mean(audio_data ** 2))
            peak = np.max(np.abs(audio_data))
            duration = len(audio_data) / sample_rate
            
            # Dynamic range
            dynamic_range_db = 20 * np.log10(peak / (rms + 1e-10))
            
            # DC offset
            dc_offset = np.mean(audio_data)
            
            # Clipping detection
            clipping_threshold = 0.99
            clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
            clipping_percentage = (clipped_samples / len(audio_data)) * 100
            
            # Silence detection
            silence_threshold = 0.001
            silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
            silence_percentage = (silent_samples / len(audio_data)) * 100
            
            return {
                'stage': stage,
                'duration': duration,
                'sample_rate': sample_rate,
                'rms': float(rms),
                'peak': float(peak),
                'dynamic_range_db': float(dynamic_range_db),
                'dc_offset': float(dc_offset),
                'clipping_percentage': float(clipping_percentage),
                'silence_percentage': float(silence_percentage),
                'samples': len(audio_data)
            }
            
        except Exception as e:
            print(f"   ⚠️  Audio analysis failed: {e}")
            return {
                'stage': stage,
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'rms': 0.0, 'peak': 0.0, 'dynamic_range_db': 0.0,
                'dc_offset': 0.0, 'clipping_percentage': 0.0,
                'silence_percentage': 0.0, 'samples': len(audio_data)
            }
    
    def _create_processing_report(self, original_metrics: Dict, processed_metrics: Dict, duration: float) -> Dict:
        """Create processing report"""
        
        # Calculate changes
        peak_change = processed_metrics['peak'] - original_metrics['peak']
        rms_change = processed_metrics['rms'] - original_metrics['rms']
        dc_change = abs(processed_metrics['dc_offset']) - abs(original_metrics['dc_offset'])
        sr_change = processed_metrics['sample_rate'] != original_metrics['sample_rate']
        
        # Generate summary
        changes = []
        if sr_change:
            changes.append(f"Sample rate: {original_metrics['sample_rate']}→{processed_metrics['sample_rate']}Hz")
        if abs(dc_change) > 0.001:
            changes.append(f"DC offset: {abs(original_metrics['dc_offset']):.4f}→{abs(processed_metrics['dc_offset']):.4f}")
        if abs(peak_change) > 0.01:
            changes.append(f"Peak: {original_metrics['peak']:.3f}→{processed_metrics['peak']:.3f}")
        if abs(rms_change) > 0.01:
            changes.append(f"RMS: {original_metrics['rms']:.3f}→{processed_metrics['rms']:.3f}")
        
        if not changes:
            summary = "Minimal changes (audio already well-formatted)"
        else:
            summary = ", ".join(changes)
        
        return {
            'original': original_metrics,
            'processed': processed_metrics,
            'duration': duration,
            'sample_rate_changed': sr_change,
            'peak_change': float(peak_change),
            'rms_change': float(rms_change),
            'dc_offset_change': float(dc_change),
            'summary': summary,
            'processing_effective': sr_change or abs(dc_change) > 0.001 or abs(peak_change) > 0.01,
            'speech_ready': True  # Always true for basic processing
        }
    
    def cleanup(self):
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
        self.cleanup()


def create_basic_preprocessing_config() -> Dict:
    """Create basic preprocessing configuration"""
    return {
        'basic': {
            'description': 'Essential audio standardization only',
            'use_case': 'Convert audio to speech recognition compatible format',
            'processing': ['dc_offset_removal', 'resampling_16khz', 'normalization', 'safe_limiting'],
            'expected_changes': 'Format standardization',
            'computational_cost': 'Very Low'
        }
    }


def main():
    """Example usage of basic audio preprocessor"""
    
    print("🔧 Basic Audio Preprocessor - Speech Recognition Ready")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = BasicAudioPreprocessor()
    
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
    
    # Ask about saving original
    save_original = input("\nSave original for comparison? (y/n, default=n): ").strip().lower() == 'y'
    
    print(f"\n🚀 Processing {selected_file.name} with basic standardization...")
    
    try:
        # Process the audio
        processed_path, original_path, report = preprocessor.process_audio(
            audio_path=selected_file,
            save_original=save_original
        )
        
        # Print detailed report
        print(f"\n📊 PROCESSING REPORT:")
        print(f"   Duration: {report['duration']:.1f}s")
        print(f"   Sample Rate Changed: {'Yes' if report['sample_rate_changed'] else 'No'}")
        print(f"   Peak Change: {report['peak_change']:+.3f}")
        print(f"   RMS Change: {report['rms_change']:+.3f}")
        print(f"   DC Offset Change: {report['dc_offset_change']:+.4f}")
        
        print(f"\n💾 FILES CREATED:")
        print(f"   Processed: {processed_path}")
        if original_path:
            print(f"   Original: {original_path}")
        
        print(f"\n💡 RESULT:")
        if report['processing_effective']:
            print(f"   ✅ Audio standardized for speech recognition engines")
        else:
            print(f"   ℹ️  Audio was already well-formatted")
        
        print(f"\n✅ Ready for: Whisper, PyAnnote, SpeechBrain, NeMo, Wav2Vec2")
        print(f"🎉 Basic audio processing complete!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()