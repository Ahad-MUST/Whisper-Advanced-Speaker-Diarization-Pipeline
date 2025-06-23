# speechbrain_engine.py - Fixed SpeechBrain Speaker Diarization Engine

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
from sklearn.metrics.pairwise import cosine_similarity
import time

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)

class SpeechBrainEngine:
    """
    SpeechBrain Speaker Diarization Engine - FIXED VERSION
    
    Uses SpeechBrain's pretrained models for:
    - Speaker embedding extraction
    - Voice Activity Detection (VAD)
    - Speaker clustering and diarization
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize SpeechBrain Engine
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.temp_files = []
        
        # SpeechBrain Configuration
        self.segment_length = 1.5  # seconds
        self.segment_shift = 0.5   # seconds
        self.embedding_model = None
        self.vad_model = None
        
        # Initialize SpeechBrain models
        self._initialize_speechbrain()
        
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for computation"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 SpeechBrain using GPU")
            else:
                device = "cpu"
                print("⚠️  SpeechBrain using CPU")
        
        return device
    
    def _initialize_speechbrain(self):
        """Initialize SpeechBrain models"""
        print(f"📥 Loading SpeechBrain models...")
        
        try:
            # Load speaker embedding model
            self._load_embedding_model()
            
            # Load VAD model if available
            self._load_vad_model()
            
            print(f"✅ SpeechBrain ready")
            
        except Exception as e:
            print(f"❌ SpeechBrain initialization failed: {e}")
            print(f"   Make sure you have installed: pip install speechbrain")
            raise
    
    def _load_embedding_model(self):
        """Load SpeechBrain speaker embedding model"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print(f"   ✅ Speaker embedding model loaded")
            
        except ImportError:
            raise ImportError("SpeechBrain not installed. Run: pip install speechbrain")
        except Exception as e:
            print(f"   ⚠️  Embedding model load failed: {e}")
            raise
    
    def _load_vad_model(self):
        """Load SpeechBrain VAD model if available"""
        try:
            from speechbrain.pretrained import VAD
            
            self.vad_model = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                run_opts={"device": self.device}
            )
            print(f"   ✅ VAD model loaded")
            
        except Exception as e:
            print(f"   ⚠️  VAD model not available: {e}")
            self.vad_model = None
    
    def _preprocess_audio_for_speechbrain(self, audio_path: Path) -> str:
        """
        Preprocess audio for SpeechBrain processing
        
        Args:
            audio_path: Original audio file path
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            print("🔧 Preprocessing audio for SpeechBrain...")
            
            # Load audio with librosa
            audio_data, sample_rate = librosa.load(
                str(audio_path), 
                sr=16000,
                mono=True,
                res_type='kaiser_fast'
            )
            
            # Ensure minimum duration
            min_duration = 1.0
            duration = len(audio_data) / 16000
            
            if duration < min_duration:
                target_length = int(min_duration * 16000)
                padding = target_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant', constant_values=0)
                print(f"   📏 Padded audio from {duration:.1f}s to {min_duration:.1f}s")
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            # Create temporary WAV file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='speechbrain_')
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
    
    def _extract_speaker_embeddings(self, audio_path: str) -> Tuple[np.ndarray, List[float], List[bool]]:
        """
        Extract speaker embeddings using SpeechBrain
        
        Returns:
            Tuple of (embeddings_matrix, frame_timestamps, voice_activity)
        """
        try:
            print("🔄 Extracting speaker embeddings with SpeechBrain...")
            
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio_data) / sr
            
            # Create sliding windows
            frame_samples = int(self.segment_length * sr)
            shift_samples = int(self.segment_shift * sr)
            
            embeddings = []
            timestamps = []
            voice_activity = []
            
            total_frames = (len(audio_data) - frame_samples) // shift_samples + 1
            
            for i in range(total_frames):
                start_sample = i * shift_samples
                end_sample = start_sample + frame_samples
                
                if end_sample > len(audio_data):
                    break
                
                frame = audio_data[start_sample:end_sample]
                start_time = start_sample / sr
                
                # Check voice activity
                has_voice = self._detect_voice_activity(frame)
                voice_activity.append(has_voice)
                
                if has_voice:
                    # Extract embedding for voiced segments
                    embedding = self._extract_embedding_from_frame(frame)
                    embeddings.append(embedding)
                else:
                    # Use zero embedding for silent segments
                    if embeddings:
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        embeddings.append(np.zeros(192))  # ECAPA-TDNN embedding size
                
                timestamps.append(start_time)
            
            if not embeddings:
                raise ValueError("No embeddings extracted from audio")
            
            embeddings_matrix = np.vstack(embeddings)
            
            print(f"   ✅ Extracted {len(embeddings)} embeddings")
            print(f"   🎙️  Voice activity: {sum(voice_activity)}/{len(voice_activity)} frames")
            
            return embeddings_matrix, timestamps, voice_activity
            
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            raise
    
    def _detect_voice_activity(self, audio_frame: np.ndarray) -> bool:
        """
        Detect voice activity in audio frame
        
        Args:
            audio_frame: Audio frame data
            
        Returns:
            True if voice is detected, False otherwise
        """
        try:
            if self.vad_model is not None:
                # Use SpeechBrain VAD model
                audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    vad_output = self.vad_model.classify_batch(audio_tensor)
                    # VAD output is typically a probability
                    voice_prob = torch.sigmoid(vad_output).item()
                    return voice_prob > 0.5
            else:
                # Simple energy-based VAD
                energy = np.mean(audio_frame ** 2)
                energy_threshold = 1e-4  # Adjust as needed
                return energy > energy_threshold
                
        except Exception:
            # Fallback: assume voice activity
            return True
    
    def _extract_embedding_from_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Extract speaker embedding from audio frame using SpeechBrain
        
        Args:
            audio_frame: Audio frame data
            
        Returns:
            Speaker embedding vector
        """
        try:
            # Convert to tensor and add batch dimension
            audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Embedding extraction failed for frame: {e}")
            # Return random embedding as fallback
            return np.random.randn(192)
    
    def _perform_speaker_clustering(
        self, 
        embeddings: np.ndarray, 
        voice_activity: List[bool],
        num_speakers: Optional[int] = None,
        max_speakers: int = 10
    ) -> np.ndarray:
        """
        Perform speaker clustering using agglomerative clustering - FIXED VERSION
        
        Args:
            embeddings: Speaker embeddings matrix
            voice_activity: Voice activity detection results
            num_speakers: Fixed number of speakers (None for auto-detection)
            max_speakers: Maximum number of speakers for auto-detection
            
        Returns:
            Cluster labels for each frame
        """
        try:
            print("🔄 Performing speaker clustering...")
            
            # Filter embeddings for voiced segments only
            voiced_embeddings = []
            voiced_indices = []
            
            for i, (embedding, has_voice) in enumerate(zip(embeddings, voice_activity)):
                if has_voice:
                    voiced_embeddings.append(embedding)
                    voiced_indices.append(i)
            
            if len(voiced_embeddings) == 0:
                print("   ⚠️  No voiced segments found, using single speaker")
                return np.zeros(len(embeddings), dtype=int)
            
            voiced_embeddings = np.array(voiced_embeddings)
            
            # Normalize embeddings
            norms = np.linalg.norm(voiced_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            voiced_embeddings_norm = voiced_embeddings / norms
            
            # Determine number of clusters
            if num_speakers is not None:
                n_clusters = num_speakers
            else:
                n_clusters = self._estimate_num_speakers(voiced_embeddings_norm, max_speakers)
            
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
            
            # FIXED: Map labels back to all frames with consecutive numbering
            all_labels = np.full(len(embeddings), -1, dtype=int)  # -1 for non-voiced
            
            for voiced_idx, label in zip(voiced_indices, voiced_labels):
                all_labels[voiced_idx] = label
            
            # Fill in non-voiced segments with nearest voiced segment label
            self._fill_nonvoiced_labels(all_labels)
            
            # FIXED: Ensure consecutive speaker numbering (0, 1, 2, ...)
            unique_labels = np.unique(all_labels[all_labels >= 0])
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            
            for i in range(len(all_labels)):
                if all_labels[i] >= 0:
                    all_labels[i] = label_mapping[all_labels[i]]
            
            n_speakers_detected = len(unique_labels)
            print(f"   ✅ Clustering complete: {n_speakers_detected} speakers detected")
            
            return all_labels
            
        except Exception as e:
            print(f"❌ Clustering failed: {e}")
            # Fallback: single speaker
            return np.zeros(len(embeddings), dtype=int)
    
    def _estimate_num_speakers(self, embeddings: np.ndarray, max_speakers: int) -> int:
        """
        Estimate optimal number of speakers using silhouette analysis
        
        Args:
            embeddings: Normalized speaker embeddings
            max_speakers: Maximum number of speakers to consider
            
        Returns:
            Estimated number of speakers
        """
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import AgglomerativeClustering
            
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
                    
                    # Calculate silhouette score
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                        
                except Exception:
                    continue
            
            return best_n_clusters
            
        except ImportError:
            # Fallback: use simple heuristic
            return min(2, max_speakers)
        except Exception:
            return 1
    
    def _fill_nonvoiced_labels(self, labels: np.ndarray):
        """
        Fill non-voiced segments (-1) with labels from nearest voiced segments
        
        Args:
            labels: Array of cluster labels with -1 for non-voiced segments
        """
        try:
            # Find voiced segments
            voiced_indices = np.where(labels >= 0)[0]
            
            if len(voiced_indices) == 0:
                labels[:] = 0
                return
            
            # Fill non-voiced segments
            for i in range(len(labels)):
                if labels[i] == -1:
                    # Find nearest voiced segment
                    distances = np.abs(voiced_indices - i)
                    nearest_idx = voiced_indices[np.argmin(distances)]
                    labels[i] = labels[nearest_idx]
                    
        except Exception:
            # Fallback: assign all to speaker 0
            labels[labels == -1] = 0
    
    def diarize_audio(
        self, 
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict:
        """
        Perform speaker diarization on audio file using SpeechBrain
        
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
        
        print(f"🎯 Diarizing with SpeechBrain: {audio_path.name}")
        
        processed_audio_path = None
        
        try:
            # Preprocess audio
            processed_audio_path = self._preprocess_audio_for_speechbrain(audio_path)
            
            # Extract speaker embeddings and voice activity
            embeddings, timestamps, voice_activity = self._extract_speaker_embeddings(processed_audio_path)
            
            # Perform speaker clustering
            cluster_labels = self._perform_speaker_clustering(
                embeddings, voice_activity, num_speakers, max_speakers
            )
            
            # Convert clusters to segments
            segments = self._clusters_to_segments(cluster_labels, timestamps)
            
            # Post-process segments
            segments = self._postprocess_segments(segments)
            
            # FIXED: Generate speaker labels based on actual speakers found
            unique_speakers = sorted(set(cluster_labels))
            speakers = [f"SPEAKER_{i:02d}" for i in range(len(unique_speakers))]
            
            # Calculate statistics
            total_duration = max([seg['end'] for seg in segments]) if segments else 0
            speaker_stats = {}
            
            # FIXED: Calculate stats for all speakers that appear in segments
            speakers_in_segments = set([seg['speaker'] for seg in segments])
            
            for speaker in speakers_in_segments:
                speaker_segments = [seg for seg in segments if seg['speaker'] == speaker]
                speaker_duration = sum([seg['duration'] for seg in speaker_segments])
                speaker_stats[speaker] = {
                    'segments': len(speaker_segments),
                    'total_duration': speaker_duration,
                    'percentage': (speaker_duration / total_duration * 100) if total_duration > 0 else 0
                }
            
            # FIXED: Update speakers list to match what's actually in the results
            speakers = sorted(list(speakers_in_segments))
            
            results = {
                'segments': segments,
                'speakers': speakers,
                'num_speakers': len(speakers),
                'total_duration': total_duration,
                'speaker_stats': speaker_stats,
                'metadata': {
                    'file_name': audio_path.name,
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers,
                    'num_speakers_detected': len(speakers),
                    'engine': 'speechbrain',
                    'embedding_model': 'spkrec-ecapa-voxceleb',
                    'segment_length': self.segment_length,
                    'segment_shift': self.segment_shift,
                    'vad_available': self.vad_model is not None,
                    'preprocessing_applied': True
                }
            }
            
            print(f"✅ SpeechBrain diarization complete!")
            print(f"   Speakers detected: {len(speakers)}")
            print(f"   Total segments: {len(segments)}")
            print(f"   Audio duration: {total_duration:.1f}s")
            print(f"   VAD model: {'✅' if self.vad_model else '❌'}")
            
            return results
            
        except Exception as e:
            print(f"❌ SpeechBrain diarization failed: {e}")
            raise
        finally:
            # Always clean up
            self._cleanup_temp_files()
    
    def _clusters_to_segments(self, cluster_labels: np.ndarray, timestamps: List[float]) -> List[Dict]:
        """Convert cluster labels and timestamps to segment format - FIXED VERSION"""
        segments = []
        
        if len(cluster_labels) == 0:
            return segments
        
        current_speaker = cluster_labels[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != current_speaker:
                # End of current segment
                segment_end = timestamps[i-1] + self.segment_length
                
                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'speaker': f"SPEAKER_{current_speaker:02d}",
                    'duration': segment_end - segment_start
                })
                
                # Start new segment
                current_speaker = cluster_labels[i]
                segment_start = timestamps[i]
        
        # Add final segment
        final_end = timestamps[-1] + self.segment_length
        segments.append({
            'start': segment_start,
            'end': final_end,
            'speaker': f"SPEAKER_{current_speaker:02d}",
            'duration': final_end - segment_start
        })
        
        return segments
    
    def _postprocess_segments(self, segments: List[Dict]) -> List[Dict]:
        """Post-process segments to remove very short segments and merge consecutive ones"""
        if not segments:
            return segments
        
        # Remove very short segments (< 0.3 seconds)
        min_duration = 0.3
        filtered_segments = [seg for seg in segments if seg['duration'] >= min_duration]
        
        if not filtered_segments:
            # If all segments were too short, keep the longest ones
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
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
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