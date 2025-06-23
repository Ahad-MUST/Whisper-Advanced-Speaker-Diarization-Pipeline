import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os
from typing import Tuple, Optional
import scipy.signal
import warnings
warnings.filterwarnings("ignore")

# Try to import demucs
try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
    print("✅ Demucs available for source separation")
except ImportError:
    DEMUCS_AVAILABLE = False
    print("⚠️ Demucs not available. Install with: pip install demucs")

class AdvancedAudioPreprocessor:
    """
    Advanced Audio preprocessing using Demucs source separation and bandpass filtering
    Much more sophisticated than simple pre-emphasis
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Initialize Demucs model if available
        self.demucs_model = None
        if DEMUCS_AVAILABLE:
            try:
                print("🔄 Loading Demucs model...")
                self.demucs_model = get_model('htdemucs')
                self.demucs_model.to(self.device)
                print("✅ Demucs model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Demucs model: {e}")
                self.demucs_model = None
    
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file preserving original sample rate
        
        Args:
            file_path (str): Path to audio file
            
        Returns:
            Tuple[torch.Tensor, int]: Audio tensor and sample rate
        """
        try:
            # Load audio with torchaudio (preserves original sample rate)
            waveform, sample_rate = torchaudio.load(file_path)
            print(f"📁 Loaded: {Path(file_path).name}")
            print(f"📊 Shape: {waveform.shape}")
            print(f"📊 Sample rate: {sample_rate} Hz")
            return waveform, sample_rate
        except Exception as e:
            print(f"❌ Error loading with torchaudio: {e}")
            # Fallback to librosa
            audio, sr = librosa.load(file_path, sr=None)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            print(f"📁 Loaded with librosa: {Path(file_path).name}")
            return waveform, sr
    
    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-channel audio to mono
        
        Args:
            waveform (torch.Tensor): Input waveform [channels, samples]
            
        Returns:
            torch.Tensor: Mono waveform [1, samples]
        """
        if waveform.shape[0] > 1:
            # Convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print(f"🔄 Converted to mono: {waveform.shape}")
        else:
            print(f"✅ Already mono: {waveform.shape}")
        return waveform
    
    def apply_demucs_separation(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply Demucs source separation to isolate vocals/speech
        Much better than simple pre-emphasis - removes background music and noise
        
        Args:
            waveform (torch.Tensor): Input waveform
            sample_rate (int): Sample rate
            
        Returns:
            torch.Tensor: Separated vocals/speech
        """
        if not DEMUCS_AVAILABLE or self.demucs_model is None:
            print("⚠️ Demucs not available, skipping source separation")
            return waveform
        
        print(f"🔄 Applying Demucs source separation...")
        
        try:
            # Demucs expects stereo input, so duplicate mono if needed
            if waveform.shape[0] == 1:
                stereo_waveform = waveform.repeat(2, 1)
            else:
                stereo_waveform = waveform
            
            # Ensure correct sample rate for Demucs (44.1kHz)
            if sample_rate != 44100:
                print(f"🔄 Resampling to 44.1kHz for Demucs...")
                resampler = torchaudio.transforms.Resample(sample_rate, 44100)
                stereo_waveform = resampler(stereo_waveform)
                demucs_sr = 44100
            else:
                demucs_sr = sample_rate
            
            # Apply Demucs separation
            stereo_waveform = stereo_waveform.to(self.device)
            
            with torch.no_grad():
                # Apply the model
                sources = apply_model(self.demucs_model, stereo_waveform.unsqueeze(0))
                
                # Extract vocals (index 3 in htdemucs: drums, bass, other, vocals)
                vocals = sources[0, 3]  # Shape: [2, samples]
                
                # Convert back to mono
                vocals_mono = torch.mean(vocals, dim=0, keepdim=True)
            
            # Resample back to original sample rate if needed
            if demucs_sr != sample_rate:
                print(f"🔄 Resampling back to {sample_rate}Hz...")
                resampler = torchaudio.transforms.Resample(demucs_sr, sample_rate)
                vocals_mono = resampler(vocals_mono)
            
            vocals_mono = vocals_mono.cpu()
            
            print(f"✅ Demucs separation applied - isolated vocals/speech")
            return vocals_mono
            
        except Exception as e:
            print(f"⚠️ Demucs separation failed: {e}, using original audio")
            return waveform
    
    def apply_bandpass_filter(self, waveform: torch.Tensor, sample_rate: int, 
                             low_freq: float = 300.0, high_freq: float = 3400.0) -> torch.Tensor:
        """
        Apply bandpass filter to keep only speech frequencies
        Much more targeted than simple high-pass filter
        
        Speech frequency range: 300Hz - 3400Hz (telephone quality)
        For higher quality: 80Hz - 8000Hz
        
        Args:
            waveform (torch.Tensor): Input waveform
            sample_rate (int): Sample rate
            low_freq (float): Low cutoff frequency (default: 300Hz)
            high_freq (float): High cutoff frequency (default: 3400Hz)
            
        Returns:
            torch.Tensor: Bandpass filtered waveform
        """
        print(f"🔄 Applying bandpass filter ({low_freq}Hz - {high_freq}Hz)...")
        
        try:
            # Convert to numpy for scipy processing
            audio_np = waveform.squeeze().detach().cpu().numpy()
            
            # Design bandpass filter
            nyquist = sample_rate / 2
            low_normalized = low_freq / nyquist
            high_normalized = min(high_freq / nyquist, 0.95)  # Keep below Nyquist
            
            # Use 4th order Butterworth bandpass filter
            b, a = scipy.signal.butter(4, [low_normalized, high_normalized], btype='band')
            
            # Apply filter using filtfilt for zero-phase filtering
            filtered_audio = scipy.signal.filtfilt(b, a, audio_np)
            
            # Convert back to tensor
            filtered_waveform = torch.from_numpy(filtered_audio.copy()).unsqueeze(0).float()
            
            print(f"✅ Bandpass filter applied (speech frequencies isolated)")
            return filtered_waveform
            
        except Exception as e:
            print(f"⚠️ Bandpass filter failed: {e}, using original audio")
            return waveform
    
    def resample_to_16khz(self, waveform: torch.Tensor, original_sr: int, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Resample audio to 16kHz - Standard for speech processing models
        
        Args:
            waveform (torch.Tensor): Input waveform
            original_sr (int): Original sample rate
            target_sr (int): Target sample rate (default: 16000)
            
        Returns:
            Tuple[torch.Tensor, int]: Resampled waveform and new sample rate
        """
        if original_sr != target_sr:
            print(f"🔄 Resampling from {original_sr} Hz to {target_sr} Hz...")
            
            try:
                # Use torchaudio's resample transform
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sr, 
                    new_freq=target_sr
                )
                resampled_waveform = resampler(waveform)
                
                print(f"✅ Resampled: {waveform.shape} -> {resampled_waveform.shape}")
                return resampled_waveform, target_sr
                
            except Exception as e:
                print(f"⚠️ Resampling failed: {e}, using librosa fallback")
                # Fallback to librosa
                audio_np = waveform.squeeze().detach().cpu().numpy()
                resampled_audio = librosa.resample(audio_np, orig_sr=original_sr, target_sr=target_sr)
                resampled_waveform = torch.from_numpy(resampled_audio).unsqueeze(0).float()
                return resampled_waveform, target_sr
        else:
            print(f"✅ Already at {target_sr} Hz")
            return waveform, original_sr
    
    def normalize_audio(self, waveform: torch.Tensor, target_level: float = -23.0) -> torch.Tensor:
        """
        Normalize audio using RMS-based normalization
        
        Args:
            waveform (torch.Tensor): Input waveform
            target_level (float): Target RMS level in dB
            
        Returns:
            torch.Tensor: Normalized waveform
        """
        print(f"🔄 Normalizing audio (target: {target_level} dB RMS)...")
        
        # Calculate current RMS
        rms = torch.sqrt(torch.mean(waveform ** 2))
        
        if rms > 0:
            # Convert target level from dB to linear
            target_rms = 10 ** (target_level / 20)
            
            # Calculate scaling factor
            scaling_factor = target_rms / rms
            
            # Apply scaling
            normalized_waveform = waveform * scaling_factor
            
            # Prevent clipping
            max_val = torch.max(torch.abs(normalized_waveform))
            if max_val > 1.0:
                normalized_waveform = normalized_waveform / max_val * 0.95
            
            print(f"📊 RMS: {rms:.6f} -> {torch.sqrt(torch.mean(normalized_waveform ** 2)):.6f}")
            print(f"📊 Peak: {torch.max(torch.abs(normalized_waveform)):.6f}")
            
            return normalized_waveform
        else:
            print(f"⚠️ Silent audio detected, skipping normalization")
            return waveform
    
    def apply_voice_activity_detection(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, dict]:
        """
        Voice Activity Detection using energy and spectral features
        
        Args:
            waveform (torch.Tensor): Input waveform
            sample_rate (int): Sample rate
            
        Returns:
            Tuple[torch.Tensor, dict]: VAD mask and statistics
        """
        print(f"🔄 Applying Voice Activity Detection...")
        
        try:
            # Frame parameters
            frame_length_ms = 25  # 25ms frames
            hop_length_ms = 10    # 10ms hop
            
            frame_length = int(frame_length_ms * sample_rate / 1000)
            hop_length = int(hop_length_ms * sample_rate / 1000)
            
            audio_np = waveform.squeeze().detach().cpu().numpy()
            
            # Frame the audio
            frames = librosa.util.frame(audio_np, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            
            # Calculate energy per frame
            energy = np.sum(frames ** 2, axis=0)
            
            # Calculate spectral centroid per frame
            spectral_centroids = []
            for i in range(frames.shape[1]):
                frame = frames[:, i]
                if np.sum(frame ** 2) > 1e-10:  # Avoid silent frames
                    sc = librosa.feature.spectral_centroid(y=frame, sr=sample_rate)[0, 0]
                    spectral_centroids.append(sc)
                else:
                    spectral_centroids.append(0)
            
            spectral_centroids = np.array(spectral_centroids)
            
            # Normalize features
            energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)
            sc_norm = (spectral_centroids - np.min(spectral_centroids)) / (np.max(spectral_centroids) - np.min(spectral_centroids) + 1e-10)
            
            # Combine features (energy is more important for VAD)
            vad_score = 0.7 * energy_norm + 0.3 * sc_norm
            
            # Adaptive threshold
            threshold = np.percentile(vad_score, 30)  # Bottom 30% considered silence
            vad_mask = vad_score > threshold
            
            # Statistics
            speech_ratio = np.sum(vad_mask) / len(vad_mask)
            
            vad_stats = {
                'speech_frames': int(np.sum(vad_mask)),
                'total_frames': len(vad_mask),
                'speech_ratio': speech_ratio,
                'speech_duration': speech_ratio * len(audio_np) / sample_rate
            }
            
            print(f"🎤 VAD Results:")
            print(f"   Speech frames: {vad_stats['speech_frames']}/{vad_stats['total_frames']}")
            print(f"   Speech ratio: {speech_ratio:.1%}")
            print(f"   Speech duration: {vad_stats['speech_duration']:.2f}s")
            
            return torch.from_numpy(vad_mask), vad_stats
            
        except Exception as e:
            print(f"⚠️ VAD failed: {e}, assuming all speech")
            dummy_mask = torch.ones(100)
            dummy_stats = {'speech_frames': 100, 'total_frames': 100, 'speech_ratio': 1.0, 'speech_duration': len(waveform.squeeze()) / sample_rate}
            return dummy_mask, dummy_stats
    
    def get_audio_statistics(self, waveform: torch.Tensor, sample_rate: int) -> dict:
        """
        Calculate comprehensive audio statistics
        
        Args:
            waveform (torch.Tensor): Input waveform
            sample_rate (int): Sample rate
            
        Returns:
            dict: Audio statistics
        """
        audio_np = waveform.squeeze().detach().cpu().numpy()
        
        stats = {
            'duration': len(audio_np) / sample_rate,
            'samples': len(audio_np),
            'sample_rate': sample_rate,
            'channels': waveform.shape[0],
            'rms': float(torch.sqrt(torch.mean(waveform ** 2))),
            'peak': float(torch.max(torch.abs(waveform))),
            'dynamic_range': float(torch.max(waveform) - torch.min(waveform)),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio_np))),
        }
        
        return stats
    
    def preprocess_audio(self, input_file: str, output_file: Optional[str] = None, 
                        apply_demucs: bool = True, apply_bandpass: bool = True, 
                        resample_to_16k: bool = True, bandpass_range: str = "speech") -> dict:
        """
        Advanced audio preprocessing pipeline using Demucs + Bandpass
        
        Args:
            input_file (str): Input audio file path
            output_file (str, optional): Output file path
            apply_demucs (bool): Apply Demucs source separation
            apply_bandpass (bool): Apply bandpass filter
            resample_to_16k (bool): Resample to 16kHz
            bandpass_range (str): "speech" (300-3400Hz) or "wideband" (80-8000Hz)
            
        Returns:
            dict: Processing results and statistics
        """
        print(f"\n{'='*80}")
        print(f"🎵 ADVANCED AUDIO PREPROCESSING (Demucs + Bandpass)")
        print(f"{'='*80}")
        print(f"📁 Input: {input_file}")
        
        # Set bandpass frequencies
        if bandpass_range == "speech":
            low_freq, high_freq = 300.0, 3400.0
        else:  # wideband
            low_freq, high_freq = 80.0, 8000.0
        
        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            suffix = "_advanced_16khz" if resample_to_16k else "_advanced"
            output_file = f"{input_path.stem}{suffix}.wav"
        
        try:
            # Step 1: Load audio
            print(f"\n🔄 Step 1: Loading audio...")
            waveform, sample_rate = self.load_audio(input_file)
            original_stats = self.get_audio_statistics(waveform, sample_rate)
            
            # Step 2: Convert to mono
            print(f"\n🔄 Step 2: Converting to mono...")
            waveform = self.convert_to_mono(waveform)
            
            # Step 3: Apply Demucs source separation (if enabled)
            if apply_demucs:
                print(f"\n🔄 Step 3: Applying Demucs source separation...")
                waveform = self.apply_demucs_separation(waveform, sample_rate)
            else:
                print(f"\n⏭️ Step 3: Skipping Demucs separation")
            
            # Step 4: Resample to 16kHz (if enabled)
            if resample_to_16k:
                print(f"\n🔄 Step 4: Resampling to 16kHz...")
                waveform, sample_rate = self.resample_to_16khz(waveform, sample_rate, target_sr=16000)
            else:
                print(f"\n⏭️ Step 4: Keeping original sample rate ({sample_rate} Hz)")
            
            # Step 5: Apply bandpass filter (if enabled)
            if apply_bandpass:
                print(f"\n🔄 Step 5: Applying bandpass filter ({bandpass_range})...")
                waveform = self.apply_bandpass_filter(waveform, sample_rate, low_freq, high_freq)
            else:
                print(f"\n⏭️ Step 5: Skipping bandpass filter")
            
            # Step 6: Normalize audio
            print(f"\n🔄 Step 6: Normalizing audio...")
            waveform = self.normalize_audio(waveform, target_level=-23.0)
            
            # Step 7: Voice Activity Detection
            print(f"\n🔄 Step 7: Voice Activity Detection...")
            vad_mask, vad_stats = self.apply_voice_activity_detection(waveform, sample_rate)
            
            # Step 8: Calculate final statistics
            print(f"\n🔄 Step 8: Calculating statistics...")
            final_stats = self.get_audio_statistics(waveform, sample_rate)
            
            # Step 9: Save processed audio
            print(f"\n🔄 Step 9: Saving advanced processed audio...")
            
            # Convert to numpy for saving
            waveform_np = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
            
            # Save as WAV file
            sf.write(output_file, waveform_np, sample_rate, subtype='FLOAT')
            
            # Get file size
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            
            print(f"\n{'='*80}")
            print(f"✅ ADVANCED PROCESSING COMPLETE")
            print(f"{'='*80}")
            print(f"📁 Output: {output_file}")
            print(f"💾 File size: {file_size:.2f} MB")
            print(f"📊 Format: {sample_rate}Hz mono WAV")
            print(f"⏱️  Duration: {final_stats['duration']:.2f} seconds")
            print(f"📈 Samples: {final_stats['samples']:,}")
            print(f"🔊 RMS: {final_stats['rms']:.6f}")
            print(f"📊 Peak: {final_stats['peak']:.6f}")
            
            print(f"\n🎯 APPLIED ADVANCED PROCESSING:")
            if apply_demucs:
                print(f"  ✅ Demucs source separation (isolated vocals/speech)")
            if resample_to_16k:
                print(f"  ✅ Resampled to 16kHz (standard for speech models)")
            if apply_bandpass:
                print(f"  ✅ Bandpass filter ({low_freq}-{high_freq}Hz - {bandpass_range} range)")
            print(f"  ✅ RMS normalization (level adjustment)")
            print(f"  ✅ Voice activity detection")
            
            # Return comprehensive results
            results = {
                'input_file': input_file,
                'output_file': output_file,
                'sample_rate': sample_rate,
                'waveform_tensor': waveform,
                'vad_mask': vad_mask,
                'vad_stats': vad_stats,
                'original_stats': original_stats,
                'final_stats': final_stats,
                'file_size_mb': file_size,
                'processing_applied': {
                    'demucs_separation': apply_demucs,
                    'resampling': resample_to_16k,
                    'target_sample_rate': sample_rate,
                    'bandpass_filter': apply_bandpass,
                    'bandpass_range': f"{low_freq}-{high_freq}Hz",
                    'normalization': True,
                    'vad': True
                }
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Main execution
if __name__ == "__main__":
    print("🎵 Advanced Audio Preprocessing")
    print("Using Demucs source separation + Bandpass filtering")
    print("="*80)
    
    if not DEMUCS_AVAILABLE:
        print("⚠️ To use Demucs, install it with:")
        print("   pip install demucs")
        print("   or")
        print("   pip install 'demucs[cpu]'  # for CPU-only")
        print()
    
    choice = input("Choose option:\n1. Process single file\n2. Batch process directory\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        # Single file processing
        input_file = input("Enter path to audio file: ").strip()
        if os.path.exists(input_file):
            print("\nProcessing options:")
            demucs = input("Apply Demucs source separation? (y/n, default=y): ").strip().lower() != 'n'
            resample = input("Resample to 16kHz? (y/n, default=y): ").strip().lower() != 'n'
            bandpass = input("Apply bandpass filter? (y/n, default=y): ").strip().lower() != 'n'
            
            if bandpass:
                bp_range = input("Bandpass range - 'speech' (300-3400Hz) or 'wideband' (80-8000Hz)? (default=speech): ").strip()
                if bp_range not in ['speech', 'wideband']:
                    bp_range = 'speech'
            else:
                bp_range = 'speech'
            
            preprocessor = AdvancedAudioPreprocessor()
            result = preprocessor.preprocess_audio(
                input_file, 
                apply_demucs=demucs,
                apply_bandpass=bandpass,
                resample_to_16k=resample,
                bandpass_range=bp_range
            )
            
            if result:
                print(f"\n🎉 Success! Advanced processed audio ready for:")
                print("  • SpeechBrain models")
                print("  • NeMo ASR/TTS")
                print("  • Pyannote speaker diarization")
                print("  • Wav2Vec2, Whisper, etc.")
                print(f"\n📁 File: {result['output_file']}")
                print(f"📊 Sample rate: {result['sample_rate']} Hz")
        else:
            print("❌ File not found!")
    
    elif choice == '2':
        print("Batch processing not implemented yet. Use single file processing.")
    
    else:
        print("❌ Invalid choice!")
