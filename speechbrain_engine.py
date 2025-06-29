# speechbrain_engine.py - Optimized SpeechBrain Speaker Diarization Engine

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
from sklearn.cluster import AgglomerativeClustering
import time

# Configure logging
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

class SpeechBrainEngine:
    """
    Optimized SpeechBrain Speaker Diarization Engine
    
    Uses SpeechBrain's models for:
    - Speaker embedding extraction
    - Voice Activity Detection (VAD)
    - Speaker clustering and diarization
    """
    
    SEGMENT_LENGTH = 1.5  # seconds
    SEGMENT_SHIFT = 0.5   # seconds
    MIN_DURATION = 1.0    # minimum audio duration
    MIN_SEGMENT_DURATION = 0.3  # minimum segment duration
    EMBEDDING_SIZE = 192  # ECAPA-TDNN embedding size
    ENERGY_THRESHOLD = 1e-4  # VAD threshold
    
    def __init__(self, device: str = "auto"):
        """
        Initialize SpeechBrain Engine
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.temp_files: List[str] = []
        self.embedding_model = None
        self.vad_model = None
        
        self._initialize_models()
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computation"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _initialize_models(self):
        """Initialize SpeechBrain models"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            print("Loading SpeechBrain models...")
            
            # Load speaker embedding model
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            
            # Try to load VAD model
            try:
                from speechbrain.pretrained import VAD
                self.vad_model = VAD.from_hparams(
                    source="speechbrain/vad-crdnn-libriparty",
                    run_opts={"device": self.device}
                )
            except Exception:
                print("VAD model not available, using energy-based detection")
                self.vad_model = None
            
            print("SpeechBrain models loaded successfully")
            
        except ImportError:
            raise ImportError("SpeechBrain not installed. Run: pip install speechbrain")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SpeechBrain models: {e}")
    
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
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            processed_audio_path = self._preprocess_audio(audio_path)
            
            # Extract embeddings and perform clustering
            embeddings, timestamps, voice_activity = self._extract_embeddings(processed_audio_path)
            cluster_labels = self._perform_clustering(embeddings, voice_activity, num_speakers, max_speakers)
            
            # Convert to segments and calculate statistics
            segments = self._create_segments(cluster_labels, timestamps)
            segments = self._postprocess_segments(segments)
            
            speakers = sorted(set(seg['speaker'] for seg in segments))
            speaker_stats = self._calculate_speaker_stats(segments)
            
            processing_time = time.time() - start_time
            total_duration = max([seg['end'] for seg in segments]) if segments else 0
            
            print(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments")
            
            return {
                'segments': segments,
                'speakers': speakers,
                'num_speakers': len(speakers),
                'total_duration': total_duration,
                'speaker_stats': speaker_stats,
                'metadata': {
                    'file_name': audio_path.name,
                    'processing_time': processing_time,
                    'num_speakers_detected': len(speakers),
                    'engine': 'speechbrain',
                    'embedding_model': 'spkrec-ecapa-voxceleb',
                    'vad_available': self.vad_model is not None
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"SpeechBrain diarization failed: {e}")
        finally:
            self._cleanup_temp_files()
    
    def _preprocess_audio(self, audio_path: Path) -> str:
        """Preprocess audio for SpeechBrain processing"""
        try:
            # Load and resample audio
            audio_data, _ = librosa.load(str(audio_path), sr=16000, mono=True, res_type='kaiser_fast')
            
            # Ensure minimum duration
            if len(audio_data) / 16000 < self.MIN_DURATION:
                target_length = int(self.MIN_DURATION * 16000)
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Save to temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='speechbrain_')
            os.close(temp_fd)
            sf.write(temp_path, audio_data, 16000, subtype='PCM_16')
            
            self.temp_files.append(temp_path)
            return temp_path
            
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def _extract_embeddings(self, audio_path: str) -> Tuple[np.ndarray, List[float], List[bool]]:
        """Extract speaker embeddings efficiently"""
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            
            # Create sliding windows
            frame_samples = int(self.SEGMENT_LENGTH * sr)
            shift_samples = int(self.SEGMENT_SHIFT * sr)
            
            embeddings = []
            timestamps = []
            voice_activity = []
            
            # Process frames
            for start_sample in range(0, len(audio_data) - frame_samples + 1, shift_samples):
                end_sample = start_sample + frame_samples
                frame = audio_data[start_sample:end_sample]
                start_time = start_sample / sr
                
                # Voice activity detection
                has_voice = self._detect_voice_activity(frame)
                voice_activity.append(has_voice)
                
                # Extract embedding
                if has_voice:
                    embedding = self._extract_embedding(frame)
                else:
                    embedding = np.zeros(self.EMBEDDING_SIZE)
                
                embeddings.append(embedding)
                timestamps.append(start_time)
            
            if not embeddings:
                raise ValueError("No embeddings extracted from audio")
            
            return np.vstack(embeddings), timestamps, voice_activity
            
        except Exception as e:
            raise RuntimeError(f"Embedding extraction failed: {e}")
    
    def _detect_voice_activity(self, audio_frame: np.ndarray) -> bool:
        """Detect voice activity in audio frame"""
        try:
            if self.vad_model is not None:
                audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    vad_output = self.vad_model.classify_batch(audio_tensor)
                    return torch.sigmoid(vad_output).item() > 0.5
            else:
                # Energy-based VAD
                return np.mean(audio_frame ** 2) > self.ENERGY_THRESHOLD
        except Exception:
            return True  # Assume voice activity on error
    
    def _extract_embedding(self, audio_frame: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio frame"""
        try:
            audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
                return embedding.squeeze().cpu().numpy()
        except Exception:
            return np.random.randn(self.EMBEDDING_SIZE) * 0.1  # Fallback
    
    def _perform_clustering(
        self, 
        embeddings: np.ndarray, 
        voice_activity: List[bool],
        num_speakers: Optional[int],
        max_speakers: int
    ) -> np.ndarray:
        """Perform speaker clustering"""
        try:
            # Filter embeddings for voiced segments
            voiced_embeddings = []
            voiced_indices = []
            
            for i, (embedding, has_voice) in enumerate(zip(embeddings, voice_activity)):
                if has_voice:
                    voiced_embeddings.append(embedding)
                    voiced_indices.append(i)
            
            if len(voiced_embeddings) == 0:
                return np.zeros(len(embeddings), dtype=int)
            
            voiced_embeddings = np.array(voiced_embeddings)
            
            # Normalize embeddings
            norms = np.linalg.norm(voiced_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            voiced_embeddings_norm = voiced_embeddings / norms
            
            # Determine number of clusters
            if num_speakers is not None:
                n_clusters = num_speakers
            else:
                n_clusters = self._estimate_speakers(voiced_embeddings_norm, max_speakers)
            
            # Perform clustering
            if n_clusters == 1 or len(voiced_embeddings) < n_clusters:
                voiced_labels = np.zeros(len(voiced_embeddings), dtype=int)
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                voiced_labels = clustering.fit_predict(voiced_embeddings_norm)
            
            # Map labels back to all frames
            all_labels = np.full(len(embeddings), -1, dtype=int)
            for voiced_idx, label in zip(voiced_indices, voiced_labels):
                all_labels[voiced_idx] = label
            
            # Fill non-voiced segments
            self._fill_nonvoiced_labels(all_labels)
            
            # Ensure consecutive numbering
            unique_labels = np.unique(all_labels[all_labels >= 0])
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            
            for i in range(len(all_labels)):
                if all_labels[i] >= 0:
                    all_labels[i] = label_mapping[all_labels[i]]
            
            return all_labels
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return np.zeros(len(embeddings), dtype=int)
    
    def _estimate_speakers(self, embeddings: np.ndarray, max_speakers: int) -> int:
        """Estimate optimal number of speakers"""
        try:
            from sklearn.metrics import silhouette_score
            
            if len(embeddings) < 4:
                return 1
            
            max_clusters = min(max_speakers, len(embeddings) // 2)
            best_score = -1
            best_n_clusters = 1
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except Exception:
                    continue
            
            return best_n_clusters
            
        except ImportError:
            return min(2, max_speakers)
        except Exception:
            return 1
    
    def _fill_nonvoiced_labels(self, labels: np.ndarray):
        """Fill non-voiced segments with nearest voiced segment labels"""
        voiced_indices = np.where(labels >= 0)[0]
        
        if len(voiced_indices) == 0:
            labels[:] = 0
            return
        
        for i in range(len(labels)):
            if labels[i] == -1:
                distances = np.abs(voiced_indices - i)
                nearest_idx = voiced_indices[np.argmin(distances)]
                labels[i] = labels[nearest_idx]
    
    def _create_segments(self, cluster_labels: np.ndarray, timestamps: List[float]) -> List[Dict]:
        """Convert cluster labels to segments"""
        if len(cluster_labels) == 0:
            return []
        
        segments = []
        current_speaker = cluster_labels[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != current_speaker:
                segment_end = timestamps[i-1] + self.SEGMENT_LENGTH
                
                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'speaker': f"SPEAKER_{current_speaker:02d}",
                    'duration': segment_end - segment_start
                })
                
                current_speaker = cluster_labels[i]
                segment_start = timestamps[i]
        
        # Add final segment
        final_end = timestamps[-1] + self.SEGMENT_LENGTH
        segments.append({
            'start': segment_start,
            'end': final_end,
            'speaker': f"SPEAKER_{current_speaker:02d}",
            'duration': final_end - segment_start
        })
        
        return segments
    
    def _postprocess_segments(self, segments: List[Dict]) -> List[Dict]:
        """Post-process segments to remove short segments and merge consecutive ones"""
        if not segments:
            return segments
        
        # Remove very short segments
        filtered_segments = [seg for seg in segments if seg['duration'] >= self.MIN_SEGMENT_DURATION]
        
        if not filtered_segments:
            segments.sort(key=lambda x: x['duration'], reverse=True)
            filtered_segments = segments[:max(1, len(segments) // 2)]
        
        # Merge consecutive segments from same speaker
        merged_segments = []
        for segment in filtered_segments:
            if (merged_segments and 
                merged_segments[-1]['speaker'] == segment['speaker'] and
                abs(merged_segments[-1]['end'] - segment['start']) < 0.2):
                # Merge with previous segment
                merged_segments[-1]['end'] = segment['end']
                merged_segments[-1]['duration'] = merged_segments[-1]['end'] - merged_segments[-1]['start']
            else:
                merged_segments.append(segment)
        
        return merged_segments
    
    def _calculate_speaker_stats(self, segments: List[Dict]) -> Dict:
        """Calculate statistics for each speaker"""
        speaker_stats = {}
        total_duration = sum(seg['duration'] for seg in segments)
        
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'total_duration': 0,
                    'percentage': 0
                }
            
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['total_duration'] += segment['duration']
        
        # Calculate percentages
        for speaker, stats in speaker_stats.items():
            stats['percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
        
        return speaker_stats
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except OSError:
                pass
        self.temp_files.clear()
    
    def get_speaker_timeline(self, results: Dict) -> List[Tuple[float, float, str]]:
        """Get simple speaker timeline"""
        timeline = []
        for segment in results['segments']:
            timeline.append((segment['start'], segment['end'], segment['speaker']))
        return sorted(timeline)
    
    def get_speaker_stats(self, results: Dict) -> Dict:
        """Get formatted speaker statistics"""
        return {
            'total_duration': results['total_duration'],
            'num_speakers': results['num_speakers'],
            'total_segments': len(results['segments']),
            'speaker_breakdown': results['speaker_stats']
        }