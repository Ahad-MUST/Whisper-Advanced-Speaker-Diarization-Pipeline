# basic_audio_preprocessor.py - Optimized Audio Preprocessor for Speech Recognition

import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

class BasicAudioPreprocessor:
    """
    Optimized Basic Audio Preprocessor for Speech Recognition Engines
    
    Performs essential audio standardization:
    - Format conversion (mono, 16kHz)
    - Normalization and DC offset removal
    - Safe limiting and clipping prevention
    """
    
    TARGET_SR = 16000
    TARGET_CHANNELS = 1
    NORMALIZATION_FACTOR = 0.95
    CLIPPING_THRESHOLD = 0.99
    SILENCE_THRESHOLD = 0.001
    
    def __init__(self):
        """Initialize basic audio preprocessor"""
        self.temp_files: List[str] = []
        
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
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Load and analyze original audio
            original_audio, original_sr = self._load_audio(audio_path)
            original_metrics = self._analyze_audio(original_audio, original_sr)
            
            # Apply processing pipeline
            processed_audio = self._apply_processing_pipeline(original_audio, original_sr)
            processed_metrics = self._analyze_audio(processed_audio, self.TARGET_SR)
            
            # Generate output path
            if output_path is None:
                output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
            
            # Save processed audio
            sf.write(output_path, processed_audio, self.TARGET_SR, subtype='PCM_16')
            
            # Save original if requested
            original_path = None
            if save_original:
                original_path = audio_path.parent / f"{audio_path.stem}_original.wav"
                sf.write(original_path, original_audio, original_sr, subtype='PCM_16')
            
            # Create processing report
            processing_report = self._create_processing_report(
                original_metrics, processed_metrics, time.time() - start_time
            )
            
            print(f"Audio preprocessing complete: {output_path.name}")
            
            return str(output_path), str(original_path) if original_path else None, processing_report
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Audio preprocessing failed: {e}")
        finally:
            self.cleanup()
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with robust channel handling"""
        try:
            audio_data, sample_rate = librosa.load(
                str(audio_path),
                sr=None,
                mono=False,
                res_type='kaiser_best'
            )
            
            # Handle multi-channel audio
            if audio_data.ndim > 1:
                audio_data = self._handle_multichannel(audio_data)
            
            return audio_data.astype(np.float32), sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Audio loading failed: {e}")
    
    def _handle_multichannel(self, audio_data: np.ndarray) -> np.ndarray:
        """Intelligently handle multi-channel audio"""
        if audio_data.shape[0] == 2:  # Stereo
            left_rms = np.sqrt(np.mean(audio_data[0] ** 2))
            right_rms = np.sqrt(np.mean(audio_data[1] ** 2))
            
            # Choose channel with stronger signal or average if similar
            if left_rms > right_rms * 1.5:
                return audio_data[0]
            elif right_rms > left_rms * 1.5:
                return audio_data[1]
            else:
                return np.mean(audio_data, axis=0)
        else:
            # Multi-channel: use first channel
            return audio_data[0]
    
    def _apply_processing_pipeline(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """Apply optimized processing pipeline"""
        processed_audio = audio_data.copy()
        
        # Remove DC offset
        processed_audio -= np.mean(processed_audio)
        
        # Resample if needed
        if original_sr != self.TARGET_SR:
            processed_audio = librosa.resample(
                processed_audio,
                orig_sr=original_sr,
                target_sr=self.TARGET_SR,
                res_type='kaiser_best'
            )
        
        # Normalize and apply safe limiting
        max_abs = np.max(np.abs(processed_audio))
        if max_abs > 1e-10:
            processed_audio = processed_audio / max_abs * self.NORMALIZATION_FACTOR
        
        # Apply soft clipping
        processed_audio = np.tanh(processed_audio * 0.95) * 0.95
        
        return processed_audio.astype(np.float32)
    
    def _analyze_audio(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze audio characteristics efficiently"""
        try:
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            peak = float(np.max(np.abs(audio_data)))
            duration = len(audio_data) / sample_rate
            dc_offset = float(np.mean(audio_data))
            
            # Calculate additional metrics
            clipped_samples = np.sum(np.abs(audio_data) > self.CLIPPING_THRESHOLD)
            clipping_percentage = (clipped_samples / len(audio_data)) * 100
            
            silent_samples = np.sum(np.abs(audio_data) < self.SILENCE_THRESHOLD)
            silence_percentage = (silent_samples / len(audio_data)) * 100
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'rms': rms,
                'peak': peak,
                'dc_offset': dc_offset,
                'clipping_percentage': float(clipping_percentage),
                'silence_percentage': float(silence_percentage),
                'samples': len(audio_data)
            }
            
        except Exception:
            # Return minimal metrics on analysis failure
            return {
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'rms': 0.0, 'peak': 0.0, 'dc_offset': 0.0,
                'clipping_percentage': 0.0, 'silence_percentage': 0.0,
                'samples': len(audio_data)
            }
    
    def _create_processing_report(
        self, 
        original_metrics: Dict, 
        processed_metrics: Dict, 
        processing_time: float
    ) -> Dict:
        """Create concise processing report"""
        
        # Calculate key changes
        sr_changed = processed_metrics['sample_rate'] != original_metrics['sample_rate']
        peak_change = processed_metrics['peak'] - original_metrics['peak']
        dc_change = abs(processed_metrics['dc_offset']) - abs(original_metrics['dc_offset'])
        
        # Generate summary
        changes = []
        if sr_changed:
            changes.append(f"Sample rate: {original_metrics['sample_rate']}→{processed_metrics['sample_rate']}Hz")
        if abs(dc_change) > 0.001:
            changes.append("DC offset removed")
        if abs(peak_change) > 0.01:
            changes.append("Audio normalized")
        
        summary = ", ".join(changes) if changes else "Minimal changes required"
        
        return {
            'original': original_metrics,
            'processed': processed_metrics,
            'processing_time': processing_time,
            'sample_rate_changed': sr_changed,
            'peak_change': float(peak_change),
            'dc_offset_change': float(dc_change),
            'summary': summary,
            'processing_effective': bool(changes),
            'speech_ready': True
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


def create_basic_preprocessing_config() -> Dict:
    """Create basic preprocessing configuration"""
    return {
        'basic': {
            'description': 'Essential audio standardization for speech recognition',
            'processing': ['dc_offset_removal', 'resampling_16khz', 'normalization', 'safe_limiting'],
            'computational_cost': 'Low'
        }
    }


def main():
    """Example usage of basic audio preprocessor"""
    print("Basic Audio Preprocessor")
    print("=" * 30)
    
    preprocessor = BasicAudioPreprocessor()
    
    # Find audio files
    audio_files = list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))
    
    if not audio_files:
        print("No audio files found")
        return
    
    print(f"Found {len(audio_files)} audio file(s)")
    
    # Process first file as example
    selected_file = audio_files[0]
    print(f"Processing: {selected_file.name}")
    
    try:
        processed_path, _, report = preprocessor.process_audio(
            audio_path=selected_file,
            save_original=False
        )
        
        print(f"Processing complete")
        print(f"Changes: {report['summary']}")
        print(f"Output: {processed_path}")
        
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()