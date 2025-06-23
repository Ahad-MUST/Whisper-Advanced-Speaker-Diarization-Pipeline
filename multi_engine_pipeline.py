# multi_engine_pipeline.py - Multi-Engine Pipeline with Clean Logging

import json
import os
import pandas as pd
import warnings
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import time

# Suppress ALL unnecessary output
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['SPEECHBRAIN_CACHE'] = '/tmp'

# Import our engines
from whisper_engine import WhisperEngine
from wav2vec2_engine import Wav2Vec2Engine
from pyannote_engine import PyAnnoteEngine
from nemo_engine import NeMoEngine
from vbx_engine import VBXEngine
from speechbrain_engine import SpeechBrainEngine

# Import basic audio preprocessor
from basic_audio_preprocessor import BasicAudioPreprocessor

class MultiEngineSpeechDiarizationPipeline:
    """
    Multi-Engine Speech Pipeline with Clean Logging
    
    ASR Engines: Whisper, Wav2Vec2
    Diarization Engines: PyAnnote, NeMo, VBX, SpeechBrain
    Features: Optional basic preprocessing, ensemble voting
    """
    
    SUPPORTED_ASR_ENGINES = {
        'whisper': 'Whisper (OpenAI)',
        'wav2vec2': 'Wav2Vec2 (Facebook)'
    }
    
    SUPPORTED_DIARIZATION_ENGINES = {
        'pyannote': 'PyAnnote',
        'nemo': 'NeMo', 
        'vbx': 'VBX',
        'speechbrain': 'SpeechBrain',
        'ensemble': 'Ensemble'
    }
    
    def __init__(
        self, 
        asr_engine: str = "whisper",
        asr_model: str = "large-v3",
        diarization_engines: Union[str, List[str]] = "pyannote", 
        device: str = "auto",
        enable_preprocessing: bool = True,
        verbose: bool = False
    ):
        """Initialize pipeline with minimal logging"""
        self.asr_engine_name = asr_engine
        self.asr_model = asr_model
        self.device = device
        self.enable_preprocessing = enable_preprocessing
        self.verbose = verbose
        
        # Validate ASR engine
        if asr_engine not in self.SUPPORTED_ASR_ENGINES:
            raise ValueError(f"Unknown ASR engine: {asr_engine}")
        
        # Handle diarization engine selection
        if isinstance(diarization_engines, str):
            if diarization_engines == "ensemble":
                self.diarization_engines = ["pyannote", "nemo", "vbx", "speechbrain"]
            else:
                self.diarization_engines = [diarization_engines]
        else:
            if "ensemble" in diarization_engines:
                self.diarization_engines = ["pyannote", "nemo", "vbx", "speechbrain"]
            else:
                self.diarization_engines = diarization_engines
        
        # Validate diarization engines
        for engine in self.diarization_engines:
            if engine not in self.SUPPORTED_DIARIZATION_ENGINES:
                raise ValueError(f"Unknown diarization engine: {engine}")
        
        # Initialize preprocessor silently
        if self.enable_preprocessing:
            try:
                self.preprocessor = BasicAudioPreprocessor()
                if self.verbose:
                    print("Preprocessing: ENABLED")
            except Exception:
                self.preprocessor = None
                self.enable_preprocessing = False
                if self.verbose:
                    print("Preprocessing: DISABLED")
        else:
            self.preprocessor = None
        
        # Initialize engines
        self.asr_engine = None
        self.loaded_diarization_engines = {}
        self.failed_diarization_engines = []
        
        if self.verbose:
            print(f"Initializing: {self.SUPPORTED_ASR_ENGINES[asr_engine]} + {len(self.diarization_engines)} diarization engines")
        
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize engines with minimal logging"""
        
        # Initialize ASR Engine
        try:
            if self.asr_engine_name == "whisper":
                self.asr_engine = WhisperEngine(model_size=self.asr_model, device=self.device)
            elif self.asr_engine_name == "wav2vec2":
                self.asr_engine = Wav2Vec2Engine(model_name=self.asr_model, device=self.device)
            
            if self.verbose:
                print(f"✓ {self.asr_engine_name.upper()} loaded")
        except Exception as e:
            print(f"✗ {self.asr_engine_name.upper()} failed: {e}")
            raise RuntimeError(f"ASR engine failed to load")
        
        # Initialize diarization engines
        loaded_count = 0
        for engine_name in self.diarization_engines:
            try:
                if engine_name == "pyannote":
                    engine = PyAnnoteEngine(device=self.device)
                elif engine_name == "nemo":
                    engine = NeMoEngine(device=self.device)
                elif engine_name == "vbx":
                    engine = VBXEngine(device=self.device)
                elif engine_name == "speechbrain":
                    engine = SpeechBrainEngine(device=self.device)
                
                self.loaded_diarization_engines[engine_name] = engine
                loaded_count += 1
                if self.verbose:
                    print(f"✓ {engine_name} loaded")
                
            except Exception as e:
                self.failed_diarization_engines.append(engine_name)
                if self.verbose:
                    print(f"✗ {engine_name} failed")
        
        if not self.loaded_diarization_engines:
            raise RuntimeError("No diarization engines could be loaded")
        
        if not self.verbose:
            print(f"Pipeline ready: {self.asr_engine_name.upper()} + {loaded_count} diarization engines")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
        apply_preprocessing: bool = True,
        enable_ensemble: bool = True
    ) -> Dict:
        """Process audio with minimal logging"""
        
        audio_path = Path(audio_path)
        print(f"Processing: {audio_path.name}")
        
        total_start_time = time.time()
        processed_audio_path = audio_path
        preprocessing_metrics = {}
        
        # Step 1: Preprocessing (silent)
        if apply_preprocessing and self.enable_preprocessing and self.preprocessor:
            try:
                processed_audio_path, _, preprocessing_metrics = self.preprocessor.process_audio(
                    audio_path=audio_path, save_original=False
                )
                if self.verbose and preprocessing_metrics.get('processing_effective'):
                    print("  Preprocessing: Applied")
            except Exception:
                processed_audio_path = audio_path
                preprocessing_metrics = {}
        
        # Step 2: ASR Transcription
        try:
            asr_results = self.asr_engine.transcribe_audio(
                audio_path=processed_audio_path,
                language=language,
                word_timestamps=True
            )
            
            detected_lang = asr_results.get('language', 'unknown')
            text_length = len(asr_results.get('text', ''))
            
            if self.verbose:
                print(f"  Transcription: {detected_lang}, {text_length} chars")
            
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            raise
        
        # Step 3: Diarization
        diarization_results = {}
        successful_engines = []
        
        for engine_name, engine in self.loaded_diarization_engines.items():
            try:
                result = engine.diarize_audio(
                    audio_path=processed_audio_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                
                diarization_results[engine_name] = result
                successful_engines.append(engine_name)
                
                if self.verbose:
                    print(f"  {engine_name}: {result['num_speakers']} speakers")
                
            except Exception as e:
                diarization_results[engine_name] = {'error': str(e)}
                if self.verbose:
                    print(f"  {engine_name}: failed")
        
        if not successful_engines:
            raise RuntimeError("All diarization engines failed")
        
        # Step 4: Choose best result
        if enable_ensemble and len(successful_engines) > 1:
            best_result = self._create_ensemble_result(diarization_results)
            ensemble_used = True
        else:
            best_result = self._choose_best_result(diarization_results)
            ensemble_used = False
        
        # Step 5: Align results
        combined_results = self._align_results(asr_results, best_result)
        
        # Add metadata
        combined_results['multi_engine'] = {
            'asr_engine': self.asr_engine_name,
            'asr_model': self.asr_model,
            'diarization_engines_used': successful_engines,
            'diarization_engines_failed': self.failed_diarization_engines,
            'ensemble_used': ensemble_used,
            'engine_results': diarization_results,
            'engine_comparison': self._compare_engine_results(diarization_results)
        }
        
        combined_results['preprocessing'] = {
            'enabled': apply_preprocessing and self.enable_preprocessing,
            'applied': apply_preprocessing and self.enable_preprocessing and 'error' not in preprocessing_metrics,
            'metrics': preprocessing_metrics,
            'processed_audio_path': str(processed_audio_path) if apply_preprocessing and self.enable_preprocessing else None
        }
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        combined_results['pipeline_metadata']['total_processing_time'] = total_time
        combined_results['pipeline_metadata']['asr_engine'] = self.asr_engine_name
        combined_results['pipeline_metadata']['asr_model'] = self.asr_model
        
        # Print final summary
        num_speakers = len(combined_results['speakers'])
        num_segments = len(combined_results['segments'])
        print(f"✓ Complete: {num_speakers} speakers, {num_segments} segments, {total_time:.1f}s")
        
        return combined_results
    
    def _compare_engine_results(self, diarization_results: Dict) -> Dict:
        """Compare engine results"""
        comparison = {'num_speakers': {}, 'success_rate': 0}
        
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        for name, result in successful_results.items():
            comparison['num_speakers'][name] = result['num_speakers']
        
        comparison['success_rate'] = len(successful_results) / len(diarization_results) * 100
        
        if successful_results:
            speaker_counts = list(comparison['num_speakers'].values())
            most_common_speakers = max(set(speaker_counts), key=speaker_counts.count)
            comparison['consensus_speakers'] = most_common_speakers
            
            agreement = speaker_counts.count(most_common_speakers) / len(speaker_counts) * 100
            comparison['speaker_agreement'] = agreement
        
        return comparison
    
    def _choose_best_result(self, diarization_results: Dict) -> Dict:
        """Choose best single result"""
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        if not successful_results:
            raise RuntimeError("All diarization engines failed")
        
        # Preference order: pyannote > speechbrain > vbx > nemo
        preference_order = ['pyannote', 'speechbrain', 'vbx', 'nemo']
        
        for preferred_engine in preference_order:
            if preferred_engine in successful_results:
                return successful_results[preferred_engine]
        
        # Use first available
        return list(successful_results.values())[0]
    
    def _create_ensemble_result(self, diarization_results: Dict) -> Dict:
        """Create ensemble result"""
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        if len(successful_results) < 2:
            return self._choose_best_result(diarization_results)
        
        # Use consensus speaker count
        speaker_counts = [result['num_speakers'] for result in successful_results.values()]
        consensus_speakers = max(set(speaker_counts), key=speaker_counts.count)
        
        # Use result with consensus speaker count
        for engine_name, result in successful_results.items():
            if result['num_speakers'] == consensus_speakers:
                result['metadata']['ensemble_method'] = 'consensus_based'
                result['metadata']['engines_combined'] = list(successful_results.keys())
                return result
        
        return self._choose_best_result(diarization_results)
    
    def _align_results(self, asr_results: Dict, diarization_results: Dict) -> Dict:
        """Align ASR and diarization results"""
        asr_segments = asr_results['segments']
        speaker_segments = diarization_results['segments']
        
        aligned_segments = []
        
        for asr_seg in asr_segments:
            asr_start, asr_end = asr_seg['start'], asr_seg['end']
            asr_text = asr_seg['text'].strip()
            
            best_speaker = self._find_speaker_for_segment(asr_start, asr_end, speaker_segments)
            
            aligned_segments.append({
                'start': asr_start,
                'end': asr_end,
                'duration': asr_end - asr_start,
                'text': asr_text,
                'speaker': best_speaker,
                'words': asr_seg.get('words', [])
            })
        
        speaker_stats = self._calculate_speaker_text_stats(aligned_segments)
        
        return {
            'segments': aligned_segments,
            'speakers': diarization_results['speakers'],
            'speaker_stats': speaker_stats,
            'asr_metadata': asr_results['metadata'],
            'diarization_metadata': diarization_results['metadata'],
            'pipeline_metadata': {
                'asr_engine': self.asr_engine_name,
                'asr_model': self.asr_model,
                'diarization_engines': self.diarization_engines,
                'total_segments': len(aligned_segments),
                'language_detected': asr_results.get('language', 'Unknown')
            }
        }
    
    def _find_speaker_for_segment(self, asr_start: float, asr_end: float, speaker_segments: List[Dict]) -> str:
        """Find best matching speaker for ASR segment"""
        best_overlap = 0
        best_speaker = "SPEAKER_UNKNOWN"
        
        for s_seg in speaker_segments:
            s_start, s_end = s_seg['start'], s_seg['end']
            
            overlap_start = max(asr_start, s_start)
            overlap_end = min(asr_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            asr_duration = asr_end - asr_start
            overlap_ratio = overlap_duration / asr_duration if asr_duration > 0 else 0
            
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = s_seg['speaker']
        
        return best_speaker
    
    def _calculate_speaker_text_stats(self, aligned_segments: List[Dict]) -> Dict:
        """Calculate speaker statistics"""
        speaker_stats = {}
        
        for segment in aligned_segments:
            speaker = segment['speaker']
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0, 'total_duration': 0, 'total_words': 0,
                    'total_characters': 0, 'text_segments': []
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
        """Save results with minimal logging"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{base_name}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save transcript
        txt_path = output_dir / f"{base_name}_transcript.txt"
        self._save_transcript(results, txt_path)
        
        # Save Excel
        excel_path = output_dir / f"{base_name}_analysis.xlsx"
        self._save_excel(results, excel_path)
        
        print(f"Results saved to: {output_dir}")
    
    def _save_transcript(self, results: Dict, output_path: Path):
        """Save clean transcript"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("SPEAKER-LABELED TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            
            asr_meta = results['asr_metadata']
            f.write(f"File: {asr_meta['file_name']}\n")
            f.write(f"Duration: {asr_meta['audio_duration_seconds']:.1f}s\n")
            f.write(f"Language: {results['pipeline_metadata'].get('language_detected', 'Unknown')}\n")
            f.write(f"Speakers: {len(results['speakers'])}\n")
            f.write(f"ASR Engine: {results['multi_engine']['asr_engine'].upper()}\n")
            f.write(f"Diarization: {', '.join(results['multi_engine']['diarization_engines_used'])}\n\n")
            
            f.write("TRANSCRIPT:\n\n")
            for segment in results['segments']:
                start_min, start_sec = divmod(segment['start'], 60)
                end_min, end_sec = divmod(segment['end'], 60)
                
                f.write(f"[{int(start_min):02d}:{int(start_sec):02d}-{int(end_min):02d}:{int(end_sec):02d}] ")
                f.write(f"{segment['speaker']}: {segment['text']}\n\n")
    
    def _save_excel(self, results: Dict, output_path: Path):
        """Save Excel analysis"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Summary
            summary_data = [
                ['File', results['asr_metadata']['file_name']],
                ['Duration (s)', results['asr_metadata']['audio_duration_seconds']],
                ['Language', results['pipeline_metadata'].get('language_detected', 'Unknown')],
                ['Speakers', len(results['speakers'])],
                ['Segments', len(results['segments'])],
                ['ASR Engine', results['multi_engine']['asr_engine'].upper()],
                ['ASR Model', results['multi_engine']['asr_model']],
                ['Processing Time (s)', results['pipeline_metadata'].get('total_processing_time', 0)]
            ]
            
            df_summary = pd.DataFrame(summary_data, columns=['Property', 'Value'])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Speaker Stats
            speaker_data = []
            for speaker, stats in results['speaker_stats'].items():
                speaker_data.append({
                    'Speaker': speaker,
                    'Duration (s)': stats['total_duration'],
                    'Duration (%)': stats['duration_percentage'],
                    'Words': stats['total_words'],
                    'Words (%)': stats['words_percentage']
                })
            
            df_speakers = pd.DataFrame(speaker_data)
            df_speakers.to_excel(writer, sheet_name='Speakers', index=False)
            
            # Segments
            segments_data = []
            for i, segment in enumerate(results['segments'], 1):
                segments_data.append({
                    'Segment': i,
                    'Start': segment['start'],
                    'End': segment['end'],
                    'Speaker': segment['speaker'],
                    'Text': segment['text']
                })
            
            df_segments = pd.DataFrame(segments_data)
            df_segments.to_excel(writer, sheet_name='Segments', index=False)
    
    def print_summary(self, results: Dict):
        """Print clean summary"""
        duration = results['asr_metadata']['audio_duration_seconds']
        language = results['pipeline_metadata'].get('language_detected', 'Unknown')
        num_speakers = len(results['speakers'])
        num_segments = len(results['segments'])
        
        print(f"\nSUMMARY:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Language: {language}")
        print(f"  Speakers: {num_speakers}")
        print(f"  Segments: {num_segments}")
        print(f"  ASR: {results['multi_engine']['asr_engine'].upper()}")
        print(f"  Diarization: {', '.join(results['multi_engine']['diarization_engines_used'])}")
        
        # Top speakers
        sorted_speakers = sorted(
            results['speaker_stats'].items(), 
            key=lambda x: x[1]['total_duration'], 
            reverse=True
        )
        
        print(f"  Top speakers:")
        for speaker, stats in sorted_speakers[:3]:
            print(f"    {speaker}: {stats['duration_percentage']:.1f}%")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'preprocessor') and self.preprocessor:
            self.preprocessor.cleanup()


# Utility functions
def find_audio_files(directory: str = ".") -> List[Path]:
    """Find audio files"""
    extensions = ["*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.ogg"]
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(ext))
        files.extend(Path(directory).glob(ext.upper()))
    return sorted(list(set(files)))


def select_audio_file() -> Optional[Path]:
    """Select audio file with minimal output"""
    files = find_audio_files()
    
    if not files:
        print("No audio files found")
        return None
    
    if len(files) == 1:
        print(f"Found: {files[0].name}")
        return files[0]
    
    print("Audio files:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    
    try:
        choice = input(f"Select (1-{len(files)}) or Enter for first: ").strip()
        if not choice:
            return files[0]
        return files[int(choice) - 1]
    except (ValueError, IndexError):
        return files[0]


def select_asr_engine() -> Tuple[str, str]:
    """Select ASR engine"""
    print("\nASR Engine:")
    print("  1. Whisper (multilingual)")
    print("  2. Wav2Vec2 (German specialist)")
    
    try:
        choice = input("Select (1-2) or Enter for Whisper: ").strip()
        if choice == "2":
            return "wav2vec2", "german"
        return "whisper", "large-v3"
    except:
        return "whisper", "large-v3"


def select_diarization_engines() -> List[str]:
    """Select diarization engines"""
    engines = ["pyannote", "nemo", "vbx", "speechbrain", "ensemble"]
    
    print("\nDiarization:")
    for i, engine in enumerate(engines, 1):
        print(f"  {i}. {engine}")
    
    try:
        choice = input("Select (1-5) or Enter for PyAnnote: ").strip()
        if not choice:
            return ["pyannote"]
        
        if ',' in choice:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            return [engines[i] for i in indices if 0 <= i < len(engines)]
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(engines):
                return [engines[idx]]
    except:
        pass
    
    return ["pyannote"]


def ask_preprocessing() -> bool:
    """Ask about preprocessing"""
    print("\nApply basic preprocessing (16kHz, normalize)?")
    try:
        choice = input("(y/n, default=y): ").strip().lower()
        return choice not in ['n', 'no']
    except:
        return True


def main():
    """Main execution with clean output"""
    try:
        print("Multi-Engine Speech Pipeline")
        print("=" * 30)
        
        # Select options
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        asr_engine, asr_model = select_asr_engine()
        diarization_engines = select_diarization_engines()
        apply_preprocessing = ask_preprocessing()
        
        print("\nInitializing...")
        
        # Initialize pipeline
        pipeline = MultiEngineSpeechDiarizationPipeline(
            asr_engine=asr_engine,
            asr_model=asr_model,
            diarization_engines=diarization_engines,
            device="auto",
            enable_preprocessing=True,
            verbose=False
        )
        
        # Process audio
        results = pipeline.process_audio(
            audio_path=audio_file,
            apply_preprocessing=apply_preprocessing,
            enable_ensemble=len(diarization_engines) > 1
        )
        
        # Save results
        output_dir = f"{asr_engine}_output"
        pipeline.save_results(results, output_dir, audio_file.stem)
        
        # Print summary
        pipeline.print_summary(results)
        
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        print("\nCancelled")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()