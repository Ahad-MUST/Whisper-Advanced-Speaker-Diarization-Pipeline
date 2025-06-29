# run.py - GDPR Compliant Whisper + SpeechBrain Speech Diarization Pipeline

import json
import os
import pandas as pd
import warnings
import time
import logging
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

# Configure logging and warnings
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Import engines
from whisper_engine import WhisperEngine
from speechbrain_engine import SpeechBrainEngine
from basic_audio_preprocessor import BasicAudioPreprocessor
from gdpr_consent_manager import GDPRConsentManager

class GDPRCompliantPipeline:
    """
    GDPR Compliant Whisper + SpeechBrain Pipeline
    
    Features:
    - Consent management
    - Data retention policies
    - Data subject rights
    - Privacy-by-design processing
    """
    
    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg"]
    
    def __init__(
        self, 
        whisper_model: str = "large-v3",
        device: str = "auto",
        enable_preprocessing: bool = True
    ):
        """Initialize GDPR compliant pipeline"""
        
        self.whisper_model = whisper_model
        self.device = device
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize GDPR manager
        self.gdpr_manager = GDPRConsentManager()
        
        # Register cleanup function
        atexit.register(self.cleanup_on_exit)
        
        # Track temporary files for GDPR compliance
        self.temp_files_created = []
        
        print("GDPR Compliant Speech Diarization Pipeline")
        print("=" * 55)
        print(f"Whisper Model: {whisper_model}")
        print(f"Device: {device}")
        print(f"Privacy: GDPR compliant processing")
        print()
        
        # Initialize engines
        self.whisper_engine = None
        self.speechbrain_engine = None
        self.preprocessor = None
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all engines with error handling"""
        try:
            # Initialize Whisper
            self.whisper_engine = WhisperEngine(
                model_size=self.whisper_model,
                device=self.device
            )
            
            # Initialize SpeechBrain
            self.speechbrain_engine = SpeechBrainEngine(device=self.device)
            
            # Initialize preprocessor if enabled
            if self.enable_preprocessing:
                try:
                    self.preprocessor = BasicAudioPreprocessor()
                    print("Audio preprocessor ready")
                except Exception as e:
                    print(f"Preprocessor initialization failed: {e}")
                    self.preprocessor = None
                    self.enable_preprocessing = False
            
            print("Pipeline initialization complete")
            print()
            
        except Exception as e:
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def request_consent_and_process(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        apply_preprocessing: bool = False
    ) -> Optional[Dict]:
        """
        GDPR compliant processing with consent management
        
        Returns None if consent not given
        """
        
        audio_path = Path(audio_path)
        
        # Step 1: Display privacy notice and collect consent
        print("GDPR COMPLIANCE: CONSENT REQUIRED")
        print("=" * 40)
        
        consent_granted = self.gdpr_manager.display_privacy_notice()
        
        if not consent_granted:
            print("Processing cannot continue without consent.")
            return None
        
        # Step 2: Record file hash for consent tracking
        file_hash = self.gdpr_manager.get_file_hash(audio_path)
        self.gdpr_manager._record_consent(granted=True, file_hash=file_hash)
        
        # Step 3: Process audio
        print("\nCONSENT GRANTED - BEGINNING PROCESSING")
        print("=" * 40)
        
        try:
            results = self.process_audio(
                audio_path=audio_path,
                language=language,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                apply_preprocessing=apply_preprocessing
            )
            
            # Step 4: Add GDPR metadata
            results['gdpr_compliance'] = {
                'consent_granted': True,
                'consent_timestamp': self.gdpr_manager.consent_file,
                'file_hash': file_hash,
                'data_retention_policy': '30 days for logs, manual deletion for outputs',
                'data_subject_rights': 'deletion, access, portability available',
                'processing_purpose': 'speech diarization and transcription',
                'legal_basis': 'consent (Article 6(1)(a) GDPR)'
            }
            
            return results
            
        except Exception as e:
            print(f"Processing failed: {e}")
            # Ensure cleanup even on failure
            self.cleanup_temp_files()
            raise
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        apply_preprocessing: bool = False
    ) -> Dict:
        """Internal processing method (consent already obtained)"""
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        file_size_mb = audio_path.stat().st_size / 1e6
        print(f"Processing: {audio_path.name} ({file_size_mb:.1f} MB)")
        print(f"Language: {language or 'Auto-detect'}")
        print(f"Preprocessing: {'Yes' if apply_preprocessing and self.enable_preprocessing else 'No'}")
        print()
        
        total_start_time = time.time()
        processed_audio_path = audio_path
        preprocessing_metrics = {}
        
        try:
            # Step 1: Optional preprocessing
            if apply_preprocessing and self.enable_preprocessing and self.preprocessor:
                processed_audio_path, preprocessing_metrics = self._apply_preprocessing(audio_path)
            
            # Step 2: Parallel processing
            transcription_results, diarization_results = self._process_parallel(
                processed_audio_path, language, num_speakers, min_speakers, max_speakers
            )
            
            # Step 3: Align and combine results
            aligned_segments = self._align_results(transcription_results, diarization_results)
            speaker_stats = self._calculate_speaker_stats(aligned_segments)
            speakers = list(speaker_stats.keys())
            
            # Create final results
            total_time = time.time() - total_start_time
            total_duration = max([seg['end'] for seg in aligned_segments]) if aligned_segments else 0
            
            results = {
                'segments': aligned_segments,
                'speakers': speakers,
                'speaker_stats': speaker_stats,
                'transcription': transcription_results,
                'diarization': diarization_results,
                'preprocessing': {
                    'applied': apply_preprocessing and self.enable_preprocessing,
                    'metrics': preprocessing_metrics
                },
                'metadata': {
                    'file_name': audio_path.name,
                    'file_size_mb': file_size_mb,
                    'total_duration': total_duration,
                    'num_speakers': len(speakers),
                    'num_segments': len(aligned_segments),
                    'language': transcription_results.get('language', 'unknown'),
                    'whisper_model': self.whisper_model,
                    'device': self.device,
                    'total_processing_time': total_time,
                    'engines': ['whisper', 'speechbrain']
                }
            }
            
            print(f"Processing complete: {total_time:.1f}s total")
            print(f"Results: {len(speakers)} speakers, {len(aligned_segments)} segments")
            print()
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {e}")
        finally:
            # GDPR: Always cleanup temporary files
            self.cleanup_temp_files()
    
    def _apply_preprocessing(self, audio_path: Path) -> Tuple[Path, Dict]:
        """Apply audio preprocessing with temp file tracking"""
        try:
            print("Applying audio preprocessing...")
            processed_path, _, metrics = self.preprocessor.process_audio(
                audio_path=audio_path,
                save_original=False
            )
            
            # Track temp file for GDPR cleanup
            self.temp_files_created.append(processed_path)
            
            if metrics.get('processing_effective'):
                print(f"Preprocessing changes: {metrics.get('summary', 'N/A')}")
            else:
                print("Audio already optimized")
            
            return Path(processed_path), metrics
            
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return audio_path, {'error': str(e)}
    
    def _process_parallel(
        self, 
        audio_path: Path, 
        language: Optional[str],
        num_speakers: Optional[int],
        min_speakers: int,
        max_speakers: int
    ) -> Tuple[Dict, Dict]:
        """Process transcription and diarization"""
        
        # Transcription
        print("Starting transcription...")
        transcription_start = time.time()
        transcription_results = self.whisper_engine.transcribe_audio(
            audio_path=audio_path,
            language=language,
            word_timestamps=True
        )
        transcription_time = time.time() - transcription_start
        
        print(f"Transcription complete: {transcription_time:.1f}s")
        print(f"Language: {transcription_results['language']}")
        print(f"Segments: {len(transcription_results['segments'])}")
        print()
        
        # Diarization
        print("Starting speaker diarization...")
        diarization_start = time.time()
        diarization_results = self.speechbrain_engine.diarize_audio(
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        diarization_time = time.time() - diarization_start
        
        print(f"Diarization complete: {diarization_time:.1f}s")
        print()
        
        # Add timing metadata
        transcription_results['metadata']['processing_time'] = transcription_time
        diarization_results['metadata']['processing_time'] = diarization_time
        
        return transcription_results, diarization_results
    
    def _align_results(self, transcription_results: Dict, diarization_results: Dict) -> List[Dict]:
        """Align transcription with speaker segments efficiently"""
        
        print("Aligning transcription with speakers...")
        
        whisper_segments = transcription_results['segments']
        speaker_segments = diarization_results['segments']
        
        aligned_segments = []
        
        for w_seg in whisper_segments:
            w_start, w_end = w_seg['start'], w_seg['end']
            w_text = w_seg['text'].strip()
            
            if not w_text:  # Skip empty segments
                continue
            
            # Find best matching speaker
            best_speaker = self._find_best_speaker(w_start, w_end, speaker_segments)
            
            aligned_segments.append({
                'start': w_start,
                'end': w_end,
                'duration': w_end - w_start,
                'text': w_text,
                'speaker': best_speaker,
                'words': w_seg.get('words', [])
            })
        
        print(f"Alignment complete: {len(aligned_segments)} final segments")
        return aligned_segments
    
    def _find_best_speaker(self, w_start: float, w_end: float, speaker_segments: List[Dict]) -> str:
        """Find best matching speaker for a transcription segment"""
        
        best_overlap = 0
        best_speaker = "SPEAKER_00"
        
        w_duration = w_end - w_start
        
        for s_seg in speaker_segments:
            s_start, s_end = s_seg['start'], s_seg['end']
            
            # Calculate overlap
            overlap_start = max(w_start, s_start)
            overlap_end = min(w_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio
            overlap_ratio = overlap_duration / w_duration if w_duration > 0 else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = s_seg['speaker']
        
        return best_speaker
    
    def _calculate_speaker_stats(self, aligned_segments: List[Dict]) -> Dict:
        """Calculate efficient speaker statistics"""
        
        speaker_stats = {}
        total_duration = sum(seg['duration'] for seg in aligned_segments)
        total_words = sum(len(seg['text'].split()) for seg in aligned_segments)
        
        for segment in aligned_segments:
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
        
        # Calculate percentages
        for speaker, stats in speaker_stats.items():
            stats['duration_percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            stats['words_percentage'] = (stats['total_words'] / total_words * 100) if total_words > 0 else 0
        
        return speaker_stats
    
    def save_results_with_gdpr_notice(self, results: Dict, output_dir: str, base_name: str):
        """Save results with GDPR compliance notice"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Saving results...")
        
        try:
            # 1. TXT file with diarization
            txt_path = output_dir / f"{base_name}.txt"
            self._save_diarization_txt(results, txt_path)
            
            # 2. JSON file with GDPR metadata
            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # 3. Excel file
            excel_path = output_dir / f"{base_name}.xlsx"
            self._save_excel_segments(results, excel_path)
            
            # 4. GDPR notice file
            gdpr_notice_path = output_dir / "GDPR_DATA_NOTICE.txt"
            self._save_gdpr_notice(results, gdpr_notice_path)
            
            print(f"Files saved to: {output_dir}")
            print(f"   {txt_path.name}")
            print(f"   {json_path.name}")
            print(f"   {excel_path.name}")
            print(f"   {gdpr_notice_path.name}")
            print()
            
            # Show data subject rights
            self._show_data_subject_rights(output_dir)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save results: {e}")
    
    def _save_gdpr_notice(self, results: Dict, output_path: Path):
        """Save GDPR compliance notice with the output"""
        
        gdpr_info = results.get('gdpr_compliance', {})
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("GDPR DATA PROCESSING NOTICE\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Data processed: {results['metadata']['file_name']}\n")
            f.write(f"Processing date: {gdpr_info.get('consent_timestamp', 'Unknown')}\n")
            f.write(f"Purpose: {gdpr_info.get('processing_purpose', 'Speech diarization')}\n")
            f.write(f"Legal basis: {gdpr_info.get('legal_basis', 'Consent')}\n")
            f.write(f"Data retention: {gdpr_info.get('data_retention_policy', 'Manual deletion')}\n")
            f.write("\n")
            f.write("YOUR RIGHTS UNDER GDPR:\n")
            f.write("• Right to deletion: Delete this folder to remove all data\n")
            f.write("• Right to access: All processed data is in this folder\n")
            f.write("• Right to portability: Copy this folder to export your data\n")
            f.write("• Right to rectification: Reprocess with corrected data\n")
            f.write("\n")
            f.write("TO EXERCISE YOUR RIGHTS:\n")
            f.write("1. Delete all files in this folder (Right to deletion)\n")
            f.write("2. Copy this folder to another location (Data portability)\n")
            f.write("3. Contact data controller for other requests\n")
            f.write("\n")
            f.write("Data Controller: [Your Organization]\n")
            f.write("Contact: [Your Contact Information]\n")
    
    def _show_data_subject_rights(self, output_dir: Path):
        """Show data subject rights options after processing"""
        
        print("GDPR DATA SUBJECT RIGHTS")
        print("=" * 30)
        print("Your data has been processed and saved.")
        print("You can exercise your rights at any time:")
        print()
        print("1. RIGHT TO DELETION: Delete all output files")
        print("2. RIGHT TO PORTABILITY: Export/copy your data")
        print("3. CONTINUE: Keep files and exit")
        print()
        
        choice = input("Enter choice (1-3, default=3): ").strip()
        
        if choice == "1":
            self.gdpr_manager.exercise_right_to_deletion(str(output_dir))
        elif choice == "2":
            export_data = self.gdpr_manager.generate_data_export(str(output_dir))
            export_file = output_dir / "data_export.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Data export saved: {export_file}")
        else:
            print("Files retained. You can delete them manually anytime.")
    
    def _save_diarization_txt(self, results: Dict, output_path: Path):
        """Save diarization transcript efficiently"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in results['segments']:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                text = segment['text']
                
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                
                f.write(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] ")
                f.write(f"{speaker}: {text}\n")
    
    def _save_excel_segments(self, results: Dict, output_path: Path):
        """Save Excel file efficiently"""
        
        segments_data = []
        for segment in results['segments']:
            start_time = segment['start']
            end_time = segment['end']
            
            # Format timestamps
            start_min, start_sec = divmod(start_time, 60)
            end_min, end_sec = divmod(end_time, 60)
            start_formatted = f"{int(start_min):02d}:{int(start_sec):02d}"
            end_formatted = f"{int(end_min):02d}:{int(end_sec):02d}"
            
            segments_data.append({
                'Speaker': segment['speaker'],
                'Start_Time': start_formatted,
                'End_Time': end_formatted,
                'Text': segment['text']
            })
        
        df = pd.DataFrame(segments_data)
        df.to_excel(output_path, index=False, engine='openpyxl')
    
    def cleanup_temp_files(self):
        """GDPR compliant cleanup of temporary files"""
        for temp_file in self.temp_files_created:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception:
                pass
        self.temp_files_created.clear()
    
    def cleanup_on_exit(self):
        """Cleanup function called on program exit"""
        self.cleanup_temp_files()
        # Clean old consent logs
        self.gdpr_manager.check_retention_policy()
    
    def print_summary(self, results: Dict):
        """Print processing summary with GDPR info"""
        
        metadata = results['metadata']
        
        print("PROCESSING SUMMARY")
        print("=" * 30)
        print(f"File: {metadata['file_name']}")
        print(f"Duration: {metadata['total_duration']:.1f}s")
        print(f"Language: {metadata['language']}")
        print(f"Speakers: {metadata['num_speakers']}")
        print(f"Segments: {metadata['num_segments']}")
        print(f"Total Time: {metadata['total_processing_time']:.1f}s")
        print(f"GDPR Compliant: Yes")
        
        print("\nSPEAKER BREAKDOWN:")
        for speaker, stats in results['speaker_stats'].items():
            print(f"   {speaker}: {stats['duration_percentage']:.1f}% ({stats['total_words']} words)")
        
        print()


def find_audio_files(directory: str = ".") -> List[Path]:
    """Find audio files efficiently"""
    audio_files = []
    directory_path = Path(directory)
    
    for ext in GDPRCompliantPipeline.SUPPORTED_EXTENSIONS:
        # Search for both lowercase and uppercase extensions
        audio_files.extend(directory_path.glob(f"*{ext}"))
        audio_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    unique_files = sorted(set(audio_files))
    return unique_files


def select_audio_file() -> Optional[Path]:
    """Let user select audio file with improved UX"""
    audio_files = find_audio_files()
    
    if not audio_files:
        print("No audio files found!")
        print(f"Supported formats: {', '.join(GDPRCompliantPipeline.SUPPORTED_EXTENSIONS)}")
        return None
    
    if len(audio_files) == 1:
        print(f"Found: {audio_files[0].name}")
        return audio_files[0]
    
    print("Available audio files:")
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
            print("\nCancelled")
            return None


def ask_preprocessing() -> bool:
    """Ask user about preprocessing with clear explanation"""
    print("\nAudio Preprocessing:")
    print("   - Converts to 16kHz mono")
    print("   - Removes DC offset and noise")
    print("   - Normalizes audio levels")
    print("   - Improves speech engine compatibility")
    print("   - Recommended: Yes")
    
    while True:
        try:
            choice = input("\nApply preprocessing? (y/n, default=y): ").strip().lower()
            
            if not choice or choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
                
        except KeyboardInterrupt:
            print("\nCancelled")
            return True


def main():
    """Main execution with GDPR compliance"""
    
    try:
        print("GDPR Compliant Speech Diarization Pipeline")
        print("=" * 50)
        
        # Select audio file
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        # Ask about preprocessing
        apply_preprocessing = ask_preprocessing()
        
        print()
        
        # Initialize GDPR compliant pipeline
        pipeline = GDPRCompliantPipeline(
            whisper_model="large-v3",
            device="auto",
            enable_preprocessing=True
        )
        
        # Process audio with consent management
        results = pipeline.request_consent_and_process(
            audio_path=audio_file,
            language=None,  # Auto-detect
            apply_preprocessing=apply_preprocessing
        )
        
        if results is None:
            print("Processing cancelled - no consent given.")
            return
        
        # Save results with GDPR compliance
        output_dir = f"output_{'with' if apply_preprocessing else 'no'}_preprocessing"
        base_name = audio_file.stem
        pipeline.save_results_with_gdpr_notice(results, output_dir, base_name)
        
        # Print summary
        pipeline.print_summary(results)
        
        print("Processing complete with GDPR compliance!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Pipeline execution failed")


if __name__ == "__main__":
    main()