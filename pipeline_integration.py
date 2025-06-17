# pipeline_integration.py - Complete Pipeline with PyAnnote + NeMo Support

import json
import os
import pandas as pd
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Suppress ALL unnecessary output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SPEECHBRAIN_CACHE'] = '/tmp'

# Import our clean backend engines
from whisper_engine import WhisperEngine
from pyannote_engine import PyAnnoteEngine
from nemo_engine import NeMoEngine

class SpeechDiarizationPipeline:
    """
    Complete pipeline combining Whisper transcription with speaker diarization
    Supports both PyAnnote and NeMo engines
    """
    
    def __init__(self, whisper_model: str = "base", diarization_engine: str = "pyannote", device: str = "auto"):
        """
        Initialize the complete pipeline
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            diarization_engine: Diarization engine ('pyannote' or 'nemo')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.whisper_model = whisper_model
        self.diarization_engine_name = diarization_engine
        self.device = device
        self.whisper_engine = None
        self.diarization_engine = None
        
        print(f"🎤 Initializing Speech Pipeline ({diarization_engine.upper()})...")
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize both engines with minimal output"""
        
        # Initialize Whisper
        print("  ⏳ Loading Whisper...")
        self.whisper_engine = WhisperEngine(
            model_size=self.whisper_model,
            device=self.device
        )
        
        # Initialize chosen diarization engine
        if self.diarization_engine_name.lower() == "pyannote":
            print("  ⏳ Loading PyAnnote...")
            self.diarization_engine = PyAnnoteEngine(device=self.device)
        elif self.diarization_engine_name.lower() == "nemo":
            print("  ⏳ Loading NeMo...")
            self.diarization_engine = NeMoEngine(device=self.device)
        else:
            raise ValueError(f"Unknown diarization engine: {self.diarization_engine_name}")
        
        print("✅ Pipeline ready!\n")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ) -> Dict:
        """Process audio file with both transcription and diarization"""
        
        audio_path = Path(audio_path)
        file_size_mb = audio_path.stat().st_size / 1e6
        
        print(f"🎯 Processing: {audio_path.name} ({file_size_mb:.1f} MB)")
        print(f"   Engine: {self.diarization_engine_name.upper()}")
        
        # Step 1: Whisper Transcription
        print("  📝 Transcribing speech...")
        whisper_results = self.whisper_engine.transcribe_audio(
            audio_path=audio_path,
            language=language,
            word_timestamps=True
        )
        
        # Step 2: Speaker Diarization
        print(f"  👥 Identifying speakers ({self.diarization_engine_name})...")
        diarization_results = self.diarization_engine.diarize_audio(
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Step 3: Align and Combine Results
        print("  🔗 Aligning transcription with speakers...")
        combined_results = self._align_results(whisper_results, diarization_results)
        
        return combined_results
    
    def _align_results(self, whisper_results: Dict, diarization_results: Dict) -> Dict:
        """Align Whisper transcription with diarization results"""
        
        # Get segments from both systems
        whisper_segments = whisper_results['segments']
        speaker_segments = diarization_results['segments']
        
        # Align each Whisper segment with speaker information
        aligned_segments = []
        
        for w_seg in whisper_segments:
            w_start, w_end = w_seg['start'], w_seg['end']
            w_text = w_seg['text'].strip()
            
            # Find best matching speaker segment
            best_speaker = self._find_speaker_for_segment(
                w_start, w_end, speaker_segments
            )
            
            # Create aligned segment
            aligned_segment = {
                'start': w_start,
                'end': w_end,
                'duration': w_end - w_start,
                'text': w_text,
                'speaker': best_speaker,
                'words': w_seg.get('words', [])
            }
            
            aligned_segments.append(aligned_segment)
        
        # Create summary statistics
        speaker_stats = self._calculate_speaker_text_stats(aligned_segments)
        
        # Combine all results
        combined_results = {
            'segments': aligned_segments,
            'speakers': diarization_results['speakers'],
            'speaker_stats': speaker_stats,
            'whisper_metadata': whisper_results['metadata'],
            'diarization_metadata': diarization_results['metadata'],
            'pipeline_metadata': {
                'whisper_model': self.whisper_model,
                'diarization_engine': self.diarization_engine_name,
                'total_segments': len(aligned_segments),
                'alignment_method': 'overlap_based'
            }
        }
        
        return combined_results
    
    def _find_speaker_for_segment(
        self, 
        w_start: float, 
        w_end: float, 
        speaker_segments: List[Dict]
    ) -> str:
        """Find the best matching speaker for a Whisper segment"""
        
        best_overlap = 0
        best_speaker = "SPEAKER_UNKNOWN"
        
        for s_seg in speaker_segments:
            s_start, s_end = s_seg['start'], s_seg['end']
            
            # Calculate overlap between segments
            overlap_start = max(w_start, s_start)
            overlap_end = min(w_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio
            whisper_duration = w_end - w_start
            overlap_ratio = overlap_duration / whisper_duration if whisper_duration > 0 else 0
            
            # Update best match if this overlap is better
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = s_seg['speaker']
        
        return best_speaker
    
    def _calculate_speaker_text_stats(self, aligned_segments: List[Dict]) -> Dict:
        """Calculate statistics for each speaker including text metrics"""
        
        speaker_stats = {}
        
        for segment in aligned_segments:
            speaker = segment['speaker']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'total_duration': 0,
                    'total_words': 0,
                    'total_characters': 0,
                    'text_segments': []
                }
            
            stats = speaker_stats[speaker]
            stats['segments'] += 1
            stats['total_duration'] += segment['duration']
            stats['total_words'] += len(segment['text'].split())
            stats['total_characters'] += len(segment['text'])
            stats['text_segments'].append(segment['text'])
        
        # Add percentages
        total_duration = sum([seg['duration'] for seg in aligned_segments])
        total_words = sum([len(seg['text'].split()) for seg in aligned_segments])
        
        for speaker, stats in speaker_stats.items():
            stats['duration_percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            stats['words_percentage'] = (stats['total_words'] / total_words * 100) if total_words > 0 else 0
        
        return speaker_stats
    
    def save_results(self, results: Dict, output_dir: str, base_name: str):
        """Save combined results in multiple formats"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("  💾 Saving results...")
        
        # 1. JSON file with complete data
        json_path = output_dir / f"{base_name}_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. Plain Whisper transcript (no speakers)
        whisper_txt_path = output_dir / f"{base_name}_whisper_transcript.txt"
        self._save_whisper_transcript(results, whisper_txt_path)
        
        # 3. Speaker-labeled transcript 
        speaker_txt_path = output_dir / f"{base_name}_speaker_transcript.txt"
        self._save_speaker_transcript(results, speaker_txt_path)
        
        # 4. Excel file with multiple sheets
        excel_path = output_dir / f"{base_name}_analysis.xlsx"
        self._save_excel_analysis(results, excel_path)
        
        # 5. RTTM file for diarization (standard format)
        rttm_path = output_dir / f"{base_name}_diarization.rttm"
        self._save_rttm_format(results, rttm_path)
    
    def _save_whisper_transcript(self, results: Dict, output_path: Path):
        """Save plain Whisper transcript as TXT (no speakers)"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("WHISPER TRANSCRIPTION\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            w_meta = results['whisper_metadata']
            
            f.write(f"File: {w_meta['file_name']}\n")
            f.write(f"Duration: {w_meta['audio_duration_seconds']:.1f}s\n")
            f.write(f"Language: {w_meta.get('language', 'Unknown')}\n")
            f.write(f"Model: {w_meta['model_size']}\n")
            f.write(f"Processing Time: {w_meta['processing_time_seconds']:.1f}s\n")
            f.write(f"Speed Ratio: {w_meta['speed_ratio']:.1f}x real-time\n")
            f.write("-" * 50 + "\n\n")
            
            # Plain transcript with timestamps (no speakers)
            f.write("TRANSCRIPT:\n\n")
            for segment in results['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                
                # Format: [MM:SS - MM:SS] Text
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                
                f.write(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] ")
                f.write(f"{text}\n\n")
    
    def _save_speaker_transcript(self, results: Dict, output_path: Path):
        """Save speaker-labeled transcript as TXT"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("SPEAKER-LABELED TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            w_meta = results['whisper_metadata']
            d_meta = results['diarization_metadata']
            p_meta = results['pipeline_metadata']
            
            f.write(f"File: {w_meta['file_name']}\n")
            f.write(f"Duration: {w_meta['audio_duration_seconds']:.1f}s\n")
            f.write(f"Language: {w_meta.get('language', 'Unknown')}\n")
            f.write(f"Speakers: {len(results['speakers'])}\n")
            f.write(f"Diarization Engine: {p_meta['diarization_engine'].upper()}\n")
            f.write(f"Processing Time: {w_meta['processing_time_seconds']:.1f}s\n")
            f.write("-" * 50 + "\n\n")
            
            # Speaker statistics
            f.write("SPEAKER SUMMARY:\n")
            for speaker, stats in results['speaker_stats'].items():
                f.write(f"{speaker}: {stats['total_duration']:.1f}s ({stats['duration_percentage']:.1f}%), ")
                f.write(f"{stats['total_words']} words ({stats['words_percentage']:.1f}%)\n")
            f.write("\n" + "-" * 50 + "\n\n")
            
            # Transcript with timestamps and speakers
            f.write("TRANSCRIPT:\n\n")
            for segment in results['segments']:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                text = segment['text']
                
                # Format: [MM:SS - MM:SS] SPEAKER: Text
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                
                f.write(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] ")
                f.write(f"{speaker}: {text}\n\n")
    
    def _save_excel_analysis(self, results: Dict, output_path: Path):
        """Save detailed analysis in Excel format"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary
            summary_data = [
                ['File Name', results['whisper_metadata']['file_name']],
                ['Duration (seconds)', results['whisper_metadata']['audio_duration_seconds']],
                ['Language', results['whisper_metadata'].get('language', 'Unknown')],
                ['Number of Speakers', len(results['speakers'])],
                ['Total Segments', len(results['segments'])],
                ['Whisper Model', results['pipeline_metadata']['whisper_model']],
                ['Diarization Engine', results['pipeline_metadata']['diarization_engine'].upper()],
                ['Processing Time (s)', results['whisper_metadata']['processing_time_seconds']],
                ['Speed Ratio', f"{results['whisper_metadata']['speed_ratio']:.2f}x"]
            ]
            df_summary = pd.DataFrame(summary_data, columns=['Property', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Speaker Statistics
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
            
            # Sheet 3: Detailed Segments
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
    
    def _save_rttm_format(self, results: Dict, output_path: Path):
        """Save diarization results in RTTM format"""
        
        with open(output_path, 'w') as f:
            filename = results['whisper_metadata']['file_name']
            
            for segment in results['segments']:
                # RTTM format: SPEAKER filename 1 start_time duration <NA> <NA> speaker_id <NA> <NA>
                f.write(f"SPEAKER {filename} 1 {segment['start']:.3f} {segment['duration']:.3f} <NA> <NA> {segment['speaker']} <NA> <NA>\n")
    
    def print_summary(self, results: Dict):
        """Print clean summary of results"""
        
        duration = results['whisper_metadata']['audio_duration_seconds']
        language = results['whisper_metadata'].get('language', 'Unknown')
        num_speakers = len(results['speakers'])
        num_segments = len(results['segments'])
        engine = results['pipeline_metadata']['diarization_engine'].upper()
        
        print(f"✅ Processing Complete!")
        print(f"   📄 Duration: {duration:.1f}s")
        print(f"   🗣️  Language: {language}")
        print(f"   👥 Speakers: {num_speakers}")
        print(f"   📝 Segments: {num_segments}")
        print(f"   🔧 Engine: {engine}")
        
        print(f"   📁 Files generated:")
        print(f"      • Whisper transcript (plain text)")
        print(f"      • Speaker transcript (with speakers)")
        print(f"      • Complete data (JSON)")
        print(f"      • Analysis (Excel)")
        print(f"      • Diarization (RTTM)")
        
        # Show top 2 speakers only
        sorted_speakers = sorted(
            results['speaker_stats'].items(), 
            key=lambda x: x[1]['total_duration'], 
            reverse=True
        )
        
        print(f"   🎙️  Speaker breakdown:")
        for speaker, stats in sorted_speakers[:2]:
            duration_pct = stats['duration_percentage']
            word_count = stats['total_words']
            print(f"      {speaker}: {duration_pct:.1f}%, {word_count} words")
        
        if len(sorted_speakers) > 2:
            print(f"      ... and {len(sorted_speakers) - 2} more speakers")


def find_audio_files(directory: str = ".") -> List[Path]:
    """Find all audio files in the specified directory"""
    audio_extensions = ["*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    audio_files = []
    
    for extension in audio_extensions:
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
    print("   1. PyAnnote (Recommended) - Fast, reliable")
    print("   2. NeMo (Advanced) - Higher accuracy, slower")
    
    while True:
        try:
            choice = input("\nSelect engine (1-2) or Enter for PyAnnote: ").strip()
            
            if not choice or choice == "1":
                return "pyannote"
            elif choice == "2":
                return "nemo"
            else:
                print("Please enter 1 or 2")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return "pyannote"  # Default


def test_setup():
    """Test if all components are properly configured"""
    try:
        import torch
        import whisper
        
        # Test PyAnnote
        try:
            from pyannote.audio import Pipeline
            from huggingface_hub import whoami
            pyannote_ok = True
        except Exception:
            pyannote_ok = False
        
        # Test NeMo
        try:
            from nemo.collections.asr.models import ClusteringDiarizer
            nemo_ok = True
        except Exception:
            nemo_ok = False
        
        if not pyannote_ok and not nemo_ok:
            return False
        
        return True
    except Exception:
        return False


def main():
    """Main pipeline execution with engine selection"""
    
    try:
        print("🎤 Speech Diarization Pipeline")
        print("=" * 40)
        
        # Quick setup check
        if not test_setup():
            print("❌ Setup incomplete. Run: python test_setup.py")
            return
        
        # Select audio file
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        # Select diarization engine
        engine = select_diarization_engine()
        
        print()  # Empty line for clarity
        
        # Initialize and run pipeline
        pipeline = SpeechDiarizationPipeline(
            whisper_model="medium", 
            diarization_engine=engine,
            device="auto"
        )
        
        # Process audio
        results = pipeline.process_audio(
            audio_path=audio_file,
            language=None,
            num_speakers=None,
            min_speakers=1,
            max_speakers=10
        )
        
        # Save results
        output_dir = "output"
        base_name = audio_file.stem
        pipeline.save_results(results, output_dir, base_name)
        
        # Print summary
        pipeline.print_summary(results)
        
        print(f"\n📁 Files saved in '{output_dir}/' directory")
        print("🎉 Done!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()