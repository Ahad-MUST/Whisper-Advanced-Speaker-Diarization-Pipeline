# whisperx_diarization_standalone.py - WhisperX + PyAnnote/SpeechBrain Integration

import os
import json
import time
import warnings
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

class WhisperXDiarizationPipeline:
    """
    Standalone WhisperX + Diarization Pipeline
    
    Combines WhisperX (large-v3) with:
    - PyAnnote speaker diarization 
    - SpeechBrain speaker diarization
    
    WhisperX advantages over Whisper:
    - Better word-level timestamps
    - Improved accuracy with alignment
    - Better handling of speech segments
    """
    
    def __init__(self, device: str = "auto", diarization_engine: str = "pyannote"):
        """
        Initialize WhisperX + Diarization Pipeline
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            diarization_engine: Diarization engine ('pyannote' or 'speechbrain')
        """
        self.device = self._setup_device(device)
        self.diarization_engine = diarization_engine
        
        # Initialize components
        self.whisperx_model = None
        self.alignment_model = None
        self.diarization_pipeline = None
        
        print(f"🎯 WhisperX + {diarization_engine.title()} Integration")
        print(f"   Device: {self.device}")
        print(f"   WhisperX Model: large-v3")
        print(f"   Diarization: {diarization_engine.title()}")
        
        self._load_models()
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device"""
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎯 Using GPU acceleration")
            else:
                device = "cpu"
                print(f"⚠️  Using CPU (GPU recommended)")
        return device
    
    def _load_models(self):
        """Load WhisperX and diarization models"""
        try:
            # Load WhisperX
            print("📥 Loading WhisperX large-v3...")
            import whisperx
            
            self.whisperx_model = whisperx.load_model(
                "large-v3", 
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            print("   ✅ WhisperX model loaded")
            
            # Load alignment model (for better word timestamps)
            print("📥 Loading alignment model...")
            self.alignment_model = whisperx.load_align_model(
                language_code="en",  # Will be updated based on detected language
                device=self.device
            )
            print("   ✅ Alignment model loaded")
            
            # Load diarization pipeline
            print(f"📥 Loading {self.diarization_engine.title()} diarization...")
            if self.diarization_engine == "pyannote":
                self._load_pyannote()
            elif self.diarization_engine == "speechbrain":
                self._load_speechbrain()
            else:
                raise ValueError(f"Unknown diarization engine: {self.diarization_engine}")
            
            print("✅ All models loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Install requirements:")
            print("   pip install whisperx")
            if self.diarization_engine == "pyannote":
                print("   pip install pyannote.audio")
            else:
                print("   pip install speechbrain")
            raise
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def _load_pyannote(self):
        """Load PyAnnote diarization pipeline"""
        try:
            from pyannote.audio import Pipeline
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
            
            if self.device == "cuda":
                import torch
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
            
            print("   ✅ PyAnnote diarization loaded")
            
        except Exception as e:
            print(f"   ❌ PyAnnote loading failed: {e}")
            print("   💡 Make sure you have HuggingFace token set up")
            print("   💡 Run: huggingface-cli login")
            raise
    
    def _load_speechbrain(self):
        """Load SpeechBrain diarization components"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
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
                print("   ✅ SpeechBrain with VAD loaded")
            except:
                self.vad_model = None
                print("   ✅ SpeechBrain embeddings loaded (no VAD)")
            
        except Exception as e:
            print(f"   ❌ SpeechBrain loading failed: {e}")
            raise
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict:
        """
        Process audio file with WhisperX + Diarization
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'de') or None for auto-detection
            num_speakers: Fixed number of speakers
            min_speakers: Minimum speakers
            max_speakers: Maximum speakers
            
        Returns:
            Complete results with aligned transcription and speakers
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"🎯 Processing: {audio_path.name}")
        print(f"   WhisperX: large-v3")
        print(f"   Diarization: {self.diarization_engine.title()}")
        
        start_time = time.time()
        
        # Step 1: Load audio
        print("📥 Loading audio...")
        import whisperx
        audio = whisperx.load_audio(str(audio_path))
        audio_duration = len(audio) / 16000
        print(f"   Duration: {audio_duration:.1f}s")
        
        # Step 2: WhisperX transcription
        print("📝 Transcribing with WhisperX...")
        transcribe_start = time.time()
        
        result = self.whisperx_model.transcribe(
            audio, 
            batch_size=16
        )
        
        # Detect language if not specified
        detected_language = result["language"]
        if language is None:
            language = detected_language
        
        transcribe_time = time.time() - transcribe_start
        print(f"   ✅ Transcription complete ({transcribe_time:.1f}s)")
        print(f"   🗣️  Language detected: {language}")
        print(f"   📝 Segments: {len(result['segments'])}")
        
        # Step 3: WhisperX alignment for better word timestamps
        print("🔗 Aligning with WhisperX...")
        align_start = time.time()
        
        # Load language-specific alignment model if needed
        if not hasattr(self, '_current_language') or self._current_language != language:
            try:
                import whisperx
                self.alignment_model = whisperx.load_align_model(
                    language_code=language,
                    device=self.device
                )
                self._current_language = language
                print(f"   📥 Loaded alignment model for {language}")
            except:
                print(f"   ⚠️  No alignment model for {language}, using default")
        
        result = whisperx.align(
            result["segments"], 
            self.alignment_model, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        align_time = time.time() - align_start
        print(f"   ✅ Alignment complete ({align_time:.1f}s)")
        
        # Step 4: Speaker diarization
        print(f"👥 Running {self.diarization_engine.title()} diarization...")
        diarization_start = time.time()
        
        if self.diarization_engine == "pyannote":
            diarization_result = self._run_pyannote_diarization(
                audio_path, num_speakers, min_speakers, max_speakers
            )
        else:
            diarization_result = self._run_speechbrain_diarization(
                audio, num_speakers, min_speakers, max_speakers
            )
        
        diarization_time = time.time() - diarization_start
        print(f"   ✅ Diarization complete ({diarization_time:.1f}s)")
        print(f"   👥 Speakers detected: {diarization_result['num_speakers']}")
        
        # Step 5: Assign speakers to WhisperX segments
        print("🔗 Assigning speakers to segments...")
        
        final_result = self._assign_speakers_to_segments(result, diarization_result)
        
        total_time = time.time() - start_time
        
        # Create comprehensive results
        comprehensive_results = {
            'segments': final_result['segments'],
            'speakers': diarization_result['speakers'],
            'num_speakers': diarization_result['num_speakers'],
            'language': language,
            'speaker_stats': self._calculate_speaker_stats(final_result['segments']),
            'metadata': {
                'file_name': audio_path.name,
                'file_size_mb': audio_path.stat().st_size / 1e6,
                'audio_duration_seconds': audio_duration,
                'language_detected': detected_language,
                'language_used': language,
                'total_processing_time': total_time,
                'transcription_time': transcribe_time,
                'alignment_time': align_time,
                'diarization_time': diarization_time,
                'whisperx_model': 'large-v3',
                'diarization_engine': self.diarization_engine,
                'device': self.device,
                'speed_ratio': audio_duration / total_time if total_time > 0 else 0
            }
        }
        
        print(f"✅ Processing complete!")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   ⚡ Speed ratio: {audio_duration / total_time:.1f}x real-time")
        
        return comprehensive_results
    
    def _run_pyannote_diarization(
        self, 
        audio_path: Path, 
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int
    ) -> Dict:
        """Run PyAnnote diarization"""
        try:
            if num_speakers:
                diarization = self.diarization_pipeline(
                    str(audio_path),
                    num_speakers=num_speakers
                )
            else:
                diarization = self.diarization_pipeline(
                    str(audio_path),
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            
            # Convert to our format
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
            
            return {
                'segments': segments,
                'speakers': list(speakers),
                'num_speakers': len(speakers),
                'engine': 'pyannote'
            }
            
        except Exception as e:
            print(f"   ❌ PyAnnote diarization failed: {e}")
            raise
    
    def _run_speechbrain_diarization(
        self, 
        audio: any,
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int
    ) -> Dict:
        """Run SpeechBrain diarization"""
        try:
            import numpy as np
            import torch
            from sklearn.cluster import AgglomerativeClustering
            
            # Parameters
            segment_length = 1.5  # seconds
            segment_shift = 0.75   # seconds
            sample_rate = 16000
            
            # Create sliding windows
            frame_samples = int(segment_length * sample_rate)
            shift_samples = int(segment_shift * sample_rate)
            
            embeddings = []
            timestamps = []
            voice_activity = []
            
            for start_sample in range(0, len(audio) - frame_samples + 1, shift_samples):
                end_sample = start_sample + frame_samples
                frame = audio[start_sample:end_sample]
                start_time = start_sample / sample_rate
                
                # Voice activity detection
                has_voice = self._detect_voice_activity_speechbrain(frame)
                voice_activity.append(has_voice)
                
                if has_voice:
                    # Extract embedding
                    embedding = self._extract_speechbrain_embedding(frame)
                    embeddings.append(embedding)
                else:
                    # Use zero embedding for silent segments
                    if embeddings:
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        embeddings.append(np.zeros(192))
                
                timestamps.append(start_time)
            
            if not embeddings:
                raise ValueError("No embeddings extracted")
            
            embeddings_matrix = np.vstack(embeddings)
            
            # Clustering
            n_clusters = num_speakers if num_speakers else self._estimate_speakers_speechbrain(
                embeddings_matrix, voice_activity, max_speakers
            )
            
            if n_clusters == 1:
                labels = np.zeros(len(embeddings), dtype=int)
            else:
                # Normalize embeddings
                norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings_norm = embeddings_matrix / norms
                
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings_norm)
            
            # Convert to segments
            segments = self._labels_to_segments_speechbrain(labels, timestamps, segment_length)
            speakers = [f"SPEAKER_{i:02d}" for i in range(n_clusters)]
            
            return {
                'segments': segments,
                'speakers': speakers,
                'num_speakers': n_clusters,
                'engine': 'speechbrain'
            }
            
        except Exception as e:
            print(f"   ❌ SpeechBrain diarization failed: {e}")
            raise
    
    def _detect_voice_activity_speechbrain(self, audio_frame: np.ndarray) -> bool:
        """Detect voice activity"""
        try:
            if self.vad_model is not None:
                import torch
                audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    vad_output = self.vad_model.classify_batch(audio_tensor)
                    voice_prob = torch.sigmoid(vad_output).item()
                    return voice_prob > 0.5
            else:
                # Simple energy-based VAD
                energy = np.mean(audio_frame ** 2)
                return energy > 1e-4
        except:
            return True
    
    def _extract_speechbrain_embedding(self, audio_frame: np.ndarray) -> np.ndarray:
        """Extract SpeechBrain embedding"""
        try:
            import torch
            audio_tensor = torch.tensor(audio_frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
                return embedding.squeeze().cpu().numpy()
        except:
            return np.random.randn(192)
    
    def _estimate_speakers_speechbrain(
        self, 
        embeddings: np.ndarray, 
        voice_activity: List[bool], 
        max_speakers: int
    ) -> int:
        """Estimate number of speakers"""
        voiced_embeddings = [emb for emb, voice in zip(embeddings, voice_activity) if voice]
        
        if len(voiced_embeddings) < 4:
            return 1
        
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import AgglomerativeClustering
            
            voiced_embeddings = np.array(voiced_embeddings)
            norms = np.linalg.norm(voiced_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            voiced_embeddings_norm = voiced_embeddings / norms
            
            best_score = -1
            best_n_clusters = 1
            
            for n_clusters in range(2, min(max_speakers + 1, len(voiced_embeddings) // 2)):
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(voiced_embeddings_norm)
                score = silhouette_score(voiced_embeddings_norm, labels, metric='cosine')
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            
            return best_n_clusters
            
        except:
            return min(2, max_speakers)
    
    def _labels_to_segments_speechbrain(
        self, 
        labels: np.ndarray, 
        timestamps: List[float], 
        segment_length: float
    ) -> List[Dict]:
        """Convert cluster labels to segments"""
        segments = []
        
        if len(labels) == 0:
            return segments
        
        current_speaker = labels[0]
        segment_start = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                segment_end = timestamps[i-1] + segment_length
                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'speaker': f"SPEAKER_{current_speaker:02d}",
                    'duration': segment_end - segment_start
                })
                current_speaker = labels[i]
                segment_start = timestamps[i]
        
        # Add final segment
        final_end = timestamps[-1] + segment_length
        segments.append({
            'start': segment_start,
            'end': final_end,
            'speaker': f"SPEAKER_{current_speaker:02d}",
            'duration': final_end - segment_start
        })
        
        return segments
    
    def _assign_speakers_to_segments(self, whisperx_result: Dict, diarization_result: Dict) -> Dict:
        """Assign speakers to WhisperX segments"""
        whisperx_segments = whisperx_result['segments']
        speaker_segments = diarization_result['segments']
        
        assigned_segments = []
        
        for segment in whisperx_segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            # Find best matching speaker
            best_speaker = self._find_best_speaker(start_time, end_time, speaker_segments)
            
            assigned_segment = {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'text': text,
                'speaker': best_speaker,
                'words': segment.get('words', [])
            }
            
            assigned_segments.append(assigned_segment)
        
        return {'segments': assigned_segments}
    
    def _find_best_speaker(self, start_time: float, end_time: float, speaker_segments: List[Dict]) -> str:
        """Find best matching speaker for a segment"""
        best_overlap = 0
        best_speaker = "SPEAKER_00"
        
        for speaker_seg in speaker_segments:
            overlap_start = max(start_time, speaker_seg['start'])
            overlap_end = min(end_time, speaker_seg['end'])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            segment_duration = end_time - start_time
            overlap_ratio = overlap_duration / segment_duration if segment_duration > 0 else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = speaker_seg['speaker']
        
        return best_speaker
    
    def _calculate_speaker_stats(self, segments: List[Dict]) -> Dict:
        """Calculate speaker statistics"""
        speaker_stats = {}
        total_duration = sum(seg['duration'] for seg in segments)
        total_words = sum(len(seg['text'].split()) for seg in segments)
        
        for segment in segments:
            speaker = segment['speaker']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'total_duration': 0,
                    'total_words': 0,
                    'total_characters': 0
                }
            
            stats = speaker_stats[speaker]
            stats['segments'] += 1
            stats['total_duration'] += segment['duration']
            stats['total_words'] += len(segment['text'].split())
            stats['total_characters'] += len(segment['text'])
        
        # Add percentages
        for speaker, stats in speaker_stats.items():
            stats['duration_percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            stats['words_percentage'] = (stats['total_words'] / total_words * 100) if total_words > 0 else 0
        
        return speaker_stats
    
    def save_results(self, results: Dict, output_dir: str, base_name: str):
        """Save results in multiple formats"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("💾 Saving results...")
        
        # 1. JSON format
        json_path = output_dir / f"{base_name}_whisperx_{self.diarization_engine}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. Transcript format
        txt_path = output_dir / f"{base_name}_whisperx_{self.diarization_engine}_transcript.txt"
        self._save_transcript(results, txt_path)
        
        # 3. CSV format
        csv_path = output_dir / f"{base_name}_whisperx_{self.diarization_engine}_segments.csv"
        self._save_csv(results, csv_path)
        
        # 4. Excel format
        excel_path = output_dir / f"{base_name}_whisperx_{self.diarization_engine}_analysis.xlsx"
        self._save_excel(results, excel_path)
        
        print(f"   📁 Files saved to {output_dir}")
        return {
            'json': str(json_path),
            'transcript': str(txt_path),
            'csv': str(csv_path),
            'excel': str(excel_path)
        }
    
    def _save_transcript(self, results: Dict, output_path: Path):
        """Save human-readable transcript"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"WHISPERX + {self.diarization_engine.upper()} TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            meta = results['metadata']
            f.write(f"File: {meta['file_name']}\n")
            f.write(f"Duration: {meta['audio_duration_seconds']:.1f}s\n")
            f.write(f"Language: {meta['language_detected']}\n")
            f.write(f"Speakers: {results['num_speakers']}\n")
            f.write(f"WhisperX Model: {meta['whisperx_model']}\n")
            f.write(f"Diarization: {meta['diarization_engine'].title()}\n")
            f.write(f"Processing Time: {meta['total_processing_time']:.1f}s\n")
            f.write(f"Speed Ratio: {meta['speed_ratio']:.1f}x real-time\n")
            f.write("\n" + "-" * 50 + "\n\n")
            
            # Speaker stats
            f.write("SPEAKER SUMMARY:\n")
            for speaker, stats in results['speaker_stats'].items():
                f.write(f"{speaker}: {stats['total_duration']:.1f}s ({stats['duration_percentage']:.1f}%), ")
                f.write(f"{stats['total_words']} words\n")
            f.write("\n" + "-" * 50 + "\n\n")
            
            # Transcript
            f.write("TRANSCRIPT:\n\n")
            for segment in results['segments']:
                start_min, start_sec = divmod(segment['start'], 60)
                end_min, end_sec = divmod(segment['end'], 60)
                
                f.write(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] ")
                f.write(f"{segment['speaker']}: {segment['text']}\n\n")
    
    def _save_csv(self, results: Dict, output_path: Path):
        """Save CSV format"""
        segments_data = []
        for i, segment in enumerate(results['segments'], 1):
            segments_data.append({
                'Segment': i,
                'Start_Time': segment['start'],
                'End_Time': segment['end'],
                'Duration': segment['duration'],
                'Speaker': segment['speaker'],
                'Word_Count': len(segment['text'].split()),
                'Text': segment['text']
            })
        
        df = pd.DataFrame(segments_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _save_excel(self, results: Dict, output_path: Path):
        """Save Excel analysis"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = [
                ['File Name', results['metadata']['file_name']],
                ['Duration (seconds)', results['metadata']['audio_duration_seconds']],
                ['Language', results['metadata']['language_detected']],
                ['Speakers', results['num_speakers']],
                ['Total Segments', len(results['segments'])],
                ['WhisperX Model', results['metadata']['whisperx_model']],
                ['Diarization Engine', results['metadata']['diarization_engine'].title()],
                ['Processing Time (s)', results['metadata']['total_processing_time']],
                ['Speed Ratio', f"{results['metadata']['speed_ratio']:.1f}x"],
                ['Device', results['metadata']['device']],
            ]
            
            df_summary = pd.DataFrame(summary_data, columns=['Property', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Speaker stats
            speaker_data = []
            for speaker, stats in results['speaker_stats'].items():
                speaker_data.append({
                    'Speaker': speaker,
                    'Segments': stats['segments'],
                    'Duration_Seconds': stats['total_duration'],
                    'Duration_Percentage': stats['duration_percentage'],
                    'Words': stats['total_words'],
                    'Words_Percentage': stats['words_percentage'],
                    'Characters': stats['total_characters']
                })
            
            df_speakers = pd.DataFrame(speaker_data)
            df_speakers.to_excel(writer, sheet_name='Speaker_Stats', index=False)
            
            # Segments
            segments_data = []
            for i, segment in enumerate(results['segments'], 1):
                segments_data.append({
                    'Segment': i,
                    'Start_Time': segment['start'],
                    'End_Time': segment['end'],
                    'Duration': segment['duration'],
                    'Speaker': segment['speaker'],
                    'Word_Count': len(segment['text'].split()),
                    'Text': segment['text']
                })
            
            df_segments = pd.DataFrame(segments_data)
            df_segments.to_excel(writer, sheet_name='Segments', index=False)


def find_audio_files(directory: str = ".") -> List[Path]:
    """Find audio files in directory"""
    extensions = ["*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.ogg"]
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(Path(directory).glob(ext))
        audio_files.extend(Path(directory).glob(ext.upper()))
    
    return sorted(set(audio_files))


def select_audio_file() -> Optional[Path]:
    """Let user select audio file"""
    audio_files = find_audio_files()
    
    if not audio_files:
        print("❌ No audio files found!")
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
            
            if not choice:
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


def select_diarization_engine() -> str:
    """Let user select diarization engine"""
    print("\n🔧 Select Diarization Engine:")
    print("   1. PyAnnote - Industry standard, reliable")
    print("   2. SpeechBrain - Speaker embeddings, good for clustering")
    
    while True:
        try:
            choice = input("\nSelect engine (1-2) or Enter for PyAnnote: ").strip()
            
            if not choice or choice == "1":
                return "pyannote"
            elif choice == "2":
                return "speechbrain"
            else:
                print("Please enter 1-2")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return "pyannote"


def main():
    """Main function"""
    print("🎯 WhisperX + Speaker Diarization Standalone")
    print("🚀 WhisperX (large-v3) + PyAnnote/SpeechBrain")
    print("=" * 50)
    
    try:
        # Select audio file
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        # Select diarization engine
        diarization_engine = select_diarization_engine()
        
        # Optional parameters
        print(f"\n⚙️  Optional Parameters:")
        try:
            language = input("Language code (e.g., 'en', 'de') or Enter for auto-detect: ").strip() or None
            num_speakers_input = input("Fixed number of speakers or Enter for auto-detect: ").strip()
            num_speakers = int(num_speakers_input) if num_speakers_input else None
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return
        except ValueError:
            num_speakers = None
        
        print()  # Empty line
        
        # Initialize pipeline
        pipeline = WhisperXDiarizationPipeline(
            device="auto",
            diarization_engine=diarization_engine
        )
        
        # Process audio
        results = pipeline.process_audio(
            audio_path=audio_file,
            language=language,
            num_speakers=num_speakers,
            min_speakers=1,
            max_speakers=10
        )
        
        # Save results
        output_dir = f"whisperx_{diarization_engine}_output"
        base_name = audio_file.stem
        saved_files = pipeline.save_results(results, output_dir, base_name)
        
        # Print summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   📄 Duration: {results['metadata']['audio_duration_seconds']:.1f}s")
        print(f"   🗣️  Language: {results['language']}")
        print(f"   👥 Speakers: {results['num_speakers']}")
        print(f"   📝 Segments: {len(results['segments'])}")
        print(f"   ⏱️  Processing: {results['metadata']['total_processing_time']:.1f}s")
        print(f"   ⚡ Speed: {results['metadata']['speed_ratio']:.1f}x real-time")
        
        # Speaker breakdown
        print(f"\n🎙️  Speaker Breakdown:")
        for speaker, stats in results['speaker_stats'].items():
            print(f"   {speaker}: {stats['duration_percentage']:.1f}%, {stats['total_words']} words")
        
        # Files saved
        print(f"\n📁 Files saved in '{output_dir}/':")
        for format_name, file_path in saved_files.items():
            print(f"   {format_name}: {Path(file_path).name}")
        
        print(f"\n🎉 WhisperX + {diarization_engine.title()} processing complete!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()