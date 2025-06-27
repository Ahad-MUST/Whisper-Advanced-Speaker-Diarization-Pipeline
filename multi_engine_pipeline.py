# multi_engine_pipeline.py - Enhanced Multi-Engine Pipeline with WhisperX Support

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
from whisperx_engine import WhisperXEngine  # NEW: WhisperX with forced alignment
from wav2vec2_engine import Wav2Vec2Engine
from pyannote_engine import PyAnnoteEngine
from nemo_engine import NeMoEngine
from vbx_engine import VBXEngine
from speechbrain_engine import SpeechBrainEngine

# Import basic audio preprocessor
from basic_audio_preprocessor import BasicAudioPreprocessor

class EnhancedMultiEngineSpeechPipeline:
    """
    Enhanced Multi-Engine Speech Pipeline with WhisperX Support
    
    ASR Engines:
    - Whisper (OpenAI) - Multilingual, robust
    - WhisperX (Enhanced) - Forced alignment + built-in diarization
    - Wav2Vec2 (Facebook) - German language specialist
    
    Diarization Engines:
    - PyAnnote, NeMo, VBX, SpeechBrain
    - WhisperX built-in diarization
    
    Features:
    - Optional basic audio preprocessing
    - Triple ASR engine support (Whisper, WhisperX, Wav2Vec2)
    - Enhanced word-level timestamps with forced alignment
    - Built-in and external diarization options
    - Ensemble voting for improved accuracy
    - German language optimization
    """
    
    SUPPORTED_ASR_ENGINES = {
        'whisper': 'Whisper (OpenAI) - Multilingual, robust',
        'whisperx': 'WhisperX (Enhanced) - Forced alignment + diarization',
        'wav2vec2': 'Wav2Vec2 (Facebook) - German specialist'
    }
    
    SUPPORTED_DIARIZATION_ENGINES = {
        'pyannote': 'PyAnnote (Recommended)',
        'nemo': 'NeMo (Advanced)', 
        'vbx': 'VBX (Clustering)',
        'speechbrain': 'SpeechBrain (Embeddings)',
        'whisperx_builtin': 'WhisperX Built-in (If WhisperX ASR selected)',
        'ensemble': 'Ensemble (All Available)'
    }
    
    def __init__(
        self, 
        asr_engine: str = "whisperx",
        asr_model: str = "large-v3",
        diarization_engines: Union[str, List[str]] = "peechbrain", 
        device: str = "auto",
        enable_preprocessing: bool = True,
        whisperx_enable_diarization: bool = True
    ):
        """
        Initialize enhanced multi-engine pipeline with WhisperX support
        
        Args:
            asr_engine: ASR engine ('whisper', 'whisperx', 'wav2vec2')
            asr_model: Model configuration:
                - For Whisper: 'tiny', 'base', 'small', 'medium', 'large', 'large-v3'
                - For WhisperX: 'tiny', 'base', 'small', 'medium', 'large', 'large-v3'
                - For Wav2Vec2: 'german', 'english', 'multilingual'
            diarization_engines: Engine(s) to use
            device: Device to use ('auto', 'cuda', 'cpu')
            enable_preprocessing: Enable basic audio preprocessing
            whisperx_enable_diarization: Enable WhisperX built-in diarization
        """
        self.asr_engine_name = asr_engine
        self.asr_model = asr_model
        self.device = device
        self.enable_preprocessing = enable_preprocessing
        self.whisperx_enable_diarization = whisperx_enable_diarization
        
        # Validate ASR engine
        if asr_engine not in self.SUPPORTED_ASR_ENGINES:
            raise ValueError(f"Unknown ASR engine: {asr_engine}. Supported: {list(self.SUPPORTED_ASR_ENGINES.keys())}")
        
        # Handle diarization engine selection
        if isinstance(diarization_engines, str):
            if diarization_engines == "ensemble":
                if asr_engine == "whisperx" and whisperx_enable_diarization:
                    self.diarization_engines = ["whisperx_builtin", "pyannote", "speechbrain"]
                else:
                    self.diarization_engines = ["pyannote", "nemo", "vbx", "speechbrain"]
            elif diarization_engines == "whisperx_builtin" and asr_engine != "whisperx":
                print("⚠️  WhisperX built-in diarization requires WhisperX ASR engine")
                self.diarization_engines = ["pyannote"]
            else:
                self.diarization_engines = [diarization_engines]
        else:
            # Handle list input
            if "ensemble" in diarization_engines:
                if asr_engine == "whisperx" and whisperx_enable_diarization:
                    self.diarization_engines = ["whisperx_builtin", "pyannote", "speechbrain"]
                else:
                    self.diarization_engines = ["pyannote", "nemo", "vbx", "speechbrain"]
            else:
                self.diarization_engines = diarization_engines
        
        # Validate diarization engines
        for engine in self.diarization_engines:
            if engine not in self.SUPPORTED_DIARIZATION_ENGINES:
                raise ValueError(f"Unknown diarization engine: {engine}. Supported: {list(self.SUPPORTED_DIARIZATION_ENGINES.keys())}")
        
        # Special handling for WhisperX built-in diarization
        if "whisperx_builtin" in self.diarization_engines and asr_engine != "whisperx":
            print("⚠️  Removing WhisperX built-in diarization (requires WhisperX ASR)")
            self.diarization_engines = [e for e in self.diarization_engines if e != "whisperx_builtin"]
            if not self.diarization_engines:
                self.diarization_engines = ["pyannote"]
        
        # Initialize basic audio preprocessing (if enabled)
        if self.enable_preprocessing:
            try:
                self.preprocessor = BasicAudioPreprocessor()
                print(f"🔧 Basic audio preprocessing: ENABLED")
                print(f"   Purpose: Format standardization for speech engines")
            except Exception as e:
                print(f"⚠️  Basic preprocessing initialization failed: {e}")
                self.preprocessor = None
                self.enable_preprocessing = False
                print(f"🔧 Basic audio preprocessing: DISABLED")
        else:
            self.preprocessor = None
            print(f"🔧 Basic audio preprocessing: DISABLED")
        
        # Initialize engines
        self.asr_engine = None
        self.loaded_diarization_engines = {}
        self.failed_diarization_engines = []
        
        print(f"🎤 Initializing Enhanced Multi-Engine Speech Pipeline")
        print(f"   ASR Engine: {self.SUPPORTED_ASR_ENGINES[asr_engine]}")
        print(f"   ASR Model: {asr_model}")
        if asr_engine == "whisperx":
            print(f"   WhisperX Features: Forced alignment + {'built-in diarization' if whisperx_enable_diarization else 'transcription only'}")
        print(f"   Diarization: {', '.join([self.SUPPORTED_DIARIZATION_ENGINES[e] for e in self.diarization_engines])}")
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize ASR and diarization engines"""
        
        # Initialize ASR Engine
        print(f"  ⏳ Loading {self.SUPPORTED_ASR_ENGINES[self.asr_engine_name]}...")
        try:
            if self.asr_engine_name == "whisper":
                self.asr_engine = WhisperEngine(
                    model_size=self.asr_model,
                    device=self.device
                )
            elif self.asr_engine_name == "whisperx":
                self.asr_engine = WhisperXEngine(
                    model_size=self.asr_model,
                    device=self.device,
                    compute_type="float16",  # Best balance
                    enable_diarization=self.whisperx_enable_diarization,
                    batch_size=16  # High performance
                )
            elif self.asr_engine_name == "wav2vec2":
                self.asr_engine = Wav2Vec2Engine(
                    model_name=self.asr_model,
                    device=self.device
                )
            else:
                raise ValueError(f"Unknown ASR engine: {self.asr_engine_name}")
            
            print(f"     ✅ {self.asr_engine_name.upper()} ready")
        except Exception as e:
            print(f"     ❌ {self.asr_engine_name.upper()} failed: {e}")
            raise RuntimeError(f"ASR engine {self.asr_engine_name} required but failed to load")
        
        # Initialize diarization engines (excluding WhisperX built-in)
        external_diarization_engines = [e for e in self.diarization_engines if e != "whisperx_builtin"]
        
        for engine_name in external_diarization_engines:
            print(f"  ⏳ Loading {self.SUPPORTED_DIARIZATION_ENGINES[engine_name]}...")
            
            try:
                if engine_name == "pyannote":
                    engine = PyAnnoteEngine(device=self.device)
                elif engine_name == "nemo":
                    engine = NeMoEngine(device=self.device)
                elif engine_name == "vbx":
                    engine = VBXEngine(device=self.device)
                elif engine_name == "speechbrain":
                    engine = SpeechBrainEngine(device=self.device)
                else:
                    raise ValueError(f"Unknown diarization engine: {engine_name}")
                
                self.loaded_diarization_engines[engine_name] = engine
                print(f"     ✅ {self.SUPPORTED_DIARIZATION_ENGINES[engine_name]} ready")
                
            except Exception as e:
                print(f"     ❌ {self.SUPPORTED_DIARIZATION_ENGINES[engine_name]} failed: {e}")
                self.failed_diarization_engines.append(engine_name)
        
        # Handle WhisperX built-in diarization
        if "whisperx_builtin" in self.diarization_engines:
            if self.asr_engine_name == "whisperx" and self.whisperx_enable_diarization:
                self.loaded_diarization_engines["whisperx_builtin"] = "built_into_asr"
                print(f"     ✅ WhisperX built-in diarization ready")
            else:
                self.failed_diarization_engines.append("whisperx_builtin")
        
        if not self.loaded_diarization_engines:
            raise RuntimeError("No diarization engines could be loaded!")
        
        print(f"✅ Enhanced Multi-Engine Pipeline ready!")
        print(f"   ASR: {self.asr_engine_name.upper()} ({self.asr_model})")
        print(f"   Diarization engines: {len(self.loaded_diarization_engines)}")
        if self.failed_diarization_engines:
            print(f"   Failed engines: {', '.join(self.failed_diarization_engines)}")
        print()
    
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
        """
        Process audio with enhanced multi-engine support including WhisperX
        
        Args:
            audio_path: Input audio file path
            language: Language for transcription (de, en, auto)
            num_speakers: Fixed number of speakers
            min_speakers: Minimum speakers
            max_speakers: Maximum speakers
            apply_preprocessing: Whether to apply basic preprocessing
            enable_ensemble: Use ensemble voting when multiple engines available
            
        Returns:
            Complete results with enhanced features and multi-engine analysis
        """
        
        audio_path = Path(audio_path)
        file_size_mb = audio_path.stat().st_size / 1e6
        
        print(f"🎯 Processing: {audio_path.name} ({file_size_mb:.1f} MB)")
        print(f"   ASR Engine: {self.asr_engine_name.upper()} ({self.asr_model})")
        print(f"   Diarization: {', '.join(self.loaded_diarization_engines.keys())}")
        print(f"   Basic Preprocessing: {'✅' if apply_preprocessing and self.enable_preprocessing else '❌'}")
        
        total_start_time = time.time()
        processed_audio_path = audio_path
        preprocessing_metrics = {}
        
        # Step 1: Optional Basic Audio Preprocessing
        if apply_preprocessing and self.enable_preprocessing and self.preprocessor:
            print("  🔧 Applying basic audio preprocessing...")
            preprocessing_start = time.time()
            
            try:
                processed_audio_path, original_comparison_path, preprocessing_metrics = self.preprocessor.process_audio(
                    audio_path=audio_path,
                    save_original=False
                )
                
                preprocessing_time = time.time() - preprocessing_start
                print(f"     ✅ Basic preprocessing completed ({preprocessing_time:.1f}s)")
                
                if 'summary' in preprocessing_metrics:
                    print(f"     📈 Changes: {preprocessing_metrics['summary']}")
                    if preprocessing_metrics.get('processing_effective'):
                        print(f"     ✅ Audio format standardized for speech engines")
                    else:
                        print(f"     ℹ️  Audio was already well-formatted")
                
            except Exception as e:
                print(f"     ⚠️  Basic preprocessing failed: {e}")
                print(f"     📄 Continuing with original audio...")
                processed_audio_path = audio_path
                preprocessing_metrics = {'error': str(e)}
        else:
            if apply_preprocessing and not self.enable_preprocessing:
                print("  ℹ️  Preprocessing requested but not available - using original audio")
            else:
                print("  📄 Using original audio (preprocessing disabled)")
        
        # Step 2: Enhanced ASR Transcription
        print(f"  📝 Transcribing speech with {self.asr_engine_name.upper()}...")
        try:
            if self.asr_engine_name == "whisperx":
                # WhisperX with enhanced features
                asr_results = self.asr_engine.transcribe_audio(
                    audio_path=processed_audio_path,
                    language=language,
                    word_timestamps=True,  # Always enable forced alignment
                    enable_diarization="whisperx_builtin" in self.loaded_diarization_engines,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            else:
                # Standard Whisper or Wav2Vec2
                asr_results = self.asr_engine.transcribe_audio(
                    audio_path=processed_audio_path,
                    language=language,
                    word_timestamps=True
                )
            
            # Show ASR-specific results
            detected_lang = asr_results.get('language', 'unknown')
            print(f"     🗣️  Language detected: {detected_lang}")
            print(f"     📝 Text length: {len(asr_results.get('text', ''))}")
            
            # Show WhisperX-specific features
            if self.asr_engine_name == "whisperx":
                metadata = asr_results.get('metadata', {})
                if metadata.get('forced_alignment_applied'):
                    print(f"     ⚡ Forced alignment applied for precise timestamps")
                if metadata.get('diarization_applied'):
                    speakers = self.asr_engine.get_speakers_list(asr_results)
                    print(f"     👥 Built-in diarization: {len(speakers)} speakers")
            
        except Exception as e:
            print(f"     ❌ {self.asr_engine_name.upper()} transcription failed: {e}")
            raise
        
        # Step 3: Multi-Engine Speaker Diarization (if external engines needed)
        external_engines = {k: v for k, v in self.loaded_diarization_engines.items() if k != "whisperx_builtin"}
        
        if external_engines:
            print(f"  👥 Running {len(external_engines)} external diarization engine(s)...")
            diarization_results = {}
            
            for engine_name, engine in external_engines.items():
                print(f"     🔄 Running {self.SUPPORTED_DIARIZATION_ENGINES[engine_name]}...")
                engine_start = time.time()
                
                try:
                    result = engine.diarize_audio(
                        audio_path=processed_audio_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    
                    engine_time = time.time() - engine_start
                    result['processing_time'] = engine_time
                    diarization_results[engine_name] = result
                    
                    print(f"        ✅ {engine_name}: {result['num_speakers']} speakers ({engine_time:.1f}s)")
                    
                except Exception as e:
                    print(f"        ❌ {engine_name} failed: {e}")
                    diarization_results[engine_name] = {'error': str(e)}
        else:
            diarization_results = {}
        
        # Step 4: Handle WhisperX built-in diarization results
        if "whisperx_builtin" in self.loaded_diarization_engines:
            print("  🔄 Extracting WhisperX built-in diarization...")
            
            # Extract speaker segments from WhisperX results
            whisperx_segments = self.asr_engine.extract_speaker_segments(asr_results)
            whisperx_speakers = self.asr_engine.get_speakers_list(asr_results)
            
            if whisperx_speakers:
                # Create diarization result format
                whisperx_diarization = {
                    'segments': whisperx_segments,
                    'speakers': whisperx_speakers,
                    'num_speakers': len(whisperx_speakers),
                    'total_duration': max([seg['end'] for seg in whisperx_segments]) if whisperx_segments else 0,
                    'speaker_stats': self._calculate_whisperx_speaker_stats(whisperx_segments),
                    'metadata': {
                        'engine': 'whisperx_builtin',
                        'integrated_with_asr': True,
                        'forced_alignment': True
                    },
                    'processing_time': 0  # Integrated with ASR
                }
                
                diarization_results['whisperx_builtin'] = whisperx_diarization
                print(f"     ✅ WhisperX built-in: {len(whisperx_speakers)} speakers (integrated)")
            else:
                print(f"     ⚠️  WhisperX built-in diarization found no speakers")
                diarization_results['whisperx_builtin'] = {'error': 'No speakers detected'}
        
        # Step 5: Engine Comparison and Ensemble
        print("  🤖 Analyzing multi-engine results...")
        
        # Choose best result or create ensemble
        if enable_ensemble and len([r for r in diarization_results.values() if 'error' not in r]) > 1:
            best_result = self._create_ensemble_result(diarization_results)
            ensemble_used = True
        else:
            best_result = self._choose_best_result(diarization_results, asr_results)
            ensemble_used = False
        
        # Step 6: Align and Combine Results
        print("  🔗 Aligning transcription with speakers...")
        combined_results = self._align_results(asr_results, best_result)
        
        # Add enhanced multi-engine metadata
        combined_results['multi_engine'] = {
            'asr_engine': self.asr_engine_name,
            'asr_model': self.asr_model,
            'asr_enhanced_features': self._get_asr_features(),
            'diarization_engines_used': list(diarization_results.keys()),
            'diarization_engines_failed': [name for name, result in diarization_results.items() if 'error' in result],
            'ensemble_used': ensemble_used,
            'engine_results': diarization_results,
            'engine_comparison': self._compare_engine_results(diarization_results),
            'whisperx_builtin_used': "whisperx_builtin" in diarization_results
        }
        
        # Add preprocessing information
        combined_results['preprocessing'] = {
            'enabled': apply_preprocessing and self.enable_preprocessing,
            'type': 'basic_standardization',
            'applied': apply_preprocessing and self.enable_preprocessing and 'error' not in preprocessing_metrics,
            'metrics': preprocessing_metrics,
            'processed_audio_path': str(processed_audio_path) if apply_preprocessing and self.enable_preprocessing else None,
            'purpose': 'Format standardization for speech recognition engines'
        }
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        combined_results['pipeline_metadata']['total_processing_time'] = total_time
        combined_results['pipeline_metadata']['multi_engine'] = True
        combined_results['pipeline_metadata']['asr_engine'] = self.asr_engine_name
        combined_results['pipeline_metadata']['asr_model'] = self.asr_model
        combined_results['pipeline_metadata']['diarization_engines_loaded'] = len(self.loaded_diarization_engines)
        combined_results['pipeline_metadata']['preprocessing_type'] = 'basic_standardization'
        combined_results['pipeline_metadata']['enhanced_features'] = self._get_asr_features()
        
        return combined_results
    
    def _get_asr_features(self) -> List[str]:
        """Get list of ASR engine features"""
        features = []
        
        if self.asr_engine_name == "whisper":
            features = ["multilingual", "word_timestamps", "robust_transcription"]
        elif self.asr_engine_name == "whisperx":
            features = ["forced_alignment", "precise_timestamps", "multilingual", "built_in_diarization"]
        elif self.asr_engine_name == "wav2vec2":
            features = ["german_optimization", "language_specialization", "word_timestamps"]
        
        return features
    
    def _calculate_whisperx_speaker_stats(self, segments: List[Dict]) -> Dict:
        """Calculate speaker statistics from WhisperX segments"""
        speaker_stats = {}
        total_duration = 0
        
        for segment in segments:
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            duration = segment.get('duration', 0)
            text = segment.get('text', '')
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'segments': 0,
                    'total_duration': 0,
                    'total_words': 0,
                    'total_characters': 0
                }
            
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['total_duration'] += duration
            speaker_stats[speaker]['total_words'] += len(text.split())
            speaker_stats[speaker]['total_characters'] += len(text)
            total_duration += duration
        
        # Add percentages
        for speaker, stats in speaker_stats.items():
            stats['percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
        
        return speaker_stats
    
    def _compare_engine_results(self, diarization_results: Dict) -> Dict:
        """Compare results from different diarization engines"""
        comparison = {
            'num_speakers': {},
            'total_segments': {},
            'processing_times': {},
            'success_rate': 0,
            'features': {}
        }
        
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        for name, result in successful_results.items():
            comparison['num_speakers'][name] = result['num_speakers']
            comparison['total_segments'][name] = len(result['segments'])
            comparison['processing_times'][name] = result.get('processing_time', 0)
            
            # Track special features
            if name == "whisperx_builtin":
                comparison['features'][name] = ["integrated_asr", "forced_alignment", "precise_timestamps"]
            else:
                comparison['features'][name] = ["external_engine", "independent_processing"]
        
        comparison['success_rate'] = len(successful_results) / len(diarization_results) * 100 if diarization_results else 0
        
        if successful_results:
            # Find consensus
            speaker_counts = list(comparison['num_speakers'].values())
            most_common_speakers = max(set(speaker_counts), key=speaker_counts.count)
            comparison['consensus_speakers'] = most_common_speakers
            
            # Agreement level
            agreement = speaker_counts.count(most_common_speakers) / len(speaker_counts) * 100
            comparison['speaker_agreement'] = agreement
        
        return comparison
    
    def _choose_best_result(self, diarization_results: Dict, asr_results: Dict) -> Dict:
        """Choose the best single result, preferring WhisperX built-in if available"""
        
        # Filter successful results
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        if not successful_results:
            # Fallback: create basic result from ASR if no diarization worked
            if self.asr_engine_name == "whisperx":
                print(f"     ⚠️  All diarization failed, using WhisperX ASR segments")
                return self._create_fallback_from_asr(asr_results)
            else:
                raise RuntimeError("All diarization engines failed!")
        
        # Preference order: whisperx_builtin > pyannote > speechbrain > vbx > nemo
        preference_order = ['whisperx_builtin', 'pyannote', 'speechbrain', 'vbx', 'nemo']
        
        for preferred_engine in preference_order:
            if preferred_engine in successful_results:
                print(f"     ✅ Using {preferred_engine} result")
                return successful_results[preferred_engine]
        
        # If no preferred engine, use first available
        engine_name = list(successful_results.keys())[0]
        print(f"     ✅ Using {engine_name} result (first available)")
        return successful_results[engine_name]
    
    def _create_fallback_from_asr(self, asr_results: Dict) -> Dict:
        """Create fallback diarization result from ASR segments"""
        segments = []
        for i, seg in enumerate(asr_results.get('segments', [])):
            segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'duration': seg.get('end', 0) - seg.get('start', 0),
                'speaker': 'SPEAKER_00',  # Single speaker fallback
                'text': seg.get('text', '')
            })
        
        return {
            'segments': segments,
            'speakers': ['SPEAKER_00'],
            'num_speakers': 1,
            'total_duration': max([seg['end'] for seg in segments]) if segments else 0,
            'speaker_stats': {'SPEAKER_00': {'segments': len(segments), 'total_duration': sum([seg['duration'] for seg in segments]), 'percentage': 100.0}},
            'metadata': {'engine': 'asr_fallback', 'note': 'Fallback from ASR when diarization failed'}
        }
    
    def _create_ensemble_result(self, diarization_results: Dict) -> Dict:
        """Create ensemble result, giving preference to WhisperX built-in"""
        
        # Filter successful results
        successful_results = {name: result for name, result in diarization_results.items() if 'error' not in result}
        
        if len(successful_results) < 2:
            return self._choose_best_result(diarization_results, {})
        
        print(f"     🤖 Creating ensemble from {len(successful_results)} engines...")
        
        # If WhisperX built-in is available and successful, prefer it for ensemble
        if 'whisperx_builtin' in successful_results:
            print(f"     ✅ Ensemble using WhisperX built-in (integrated + precise)")
            result = successful_results['whisperx_builtin']
            result['metadata']['ensemble_method'] = 'whisperx_preferred'
            result['metadata']['engines_combined'] = list(successful_results.keys())
            return result
        
        # Otherwise use consensus approach
        speaker_counts = [result['num_speakers'] for result in successful_results.values()]
        consensus_speakers = max(set(speaker_counts), key=speaker_counts.count)
        
        # Use result with consensus speaker count
        for engine_name, result in successful_results.items():
            if result['num_speakers'] == consensus_speakers:
                print(f"     ✅ Ensemble using {engine_name} (consensus: {consensus_speakers} speakers)")
                result['metadata']['ensemble_method'] = 'consensus_based'
                result['metadata']['engines_combined'] = list(successful_results.keys())
                return result
        
        # Fallback to best single result
        return self._choose_best_result(diarization_results, {})
    
    def _align_results(self, asr_results: Dict, diarization_results: Dict) -> Dict:
        """Align ASR transcription with diarization results, handling WhisperX integration"""
        
        # If using WhisperX with built-in diarization, segments are already aligned
        if (self.asr_engine_name == "whisperx" and 
            diarization_results.get('metadata', {}).get('engine') == 'whisperx_builtin'):
            
            print("     ✅ Using pre-aligned WhisperX segments")
            
            # Extract aligned segments directly from ASR results
            aligned_segments = self.asr_engine.extract_speaker_segments(asr_results)
            speakers = self.asr_engine.get_speakers_list(asr_results)
            
            if not speakers:  # Fallback if no speakers detected
                speakers = ['SPEAKER_00']
                for segment in aligned_segments:
                    segment['speaker'] = 'SPEAKER_00'
            
            speaker_stats = self._calculate_speaker_text_stats(aligned_segments)
            
        else:
            # Standard alignment process for external diarization engines
            asr_segments = asr_results['segments']
            speaker_segments = diarization_results['segments']
            
            aligned_segments = []
            
            for asr_seg in asr_segments:
                asr_start, asr_end = asr_seg['start'], asr_seg['end']
                asr_text = asr_seg['text'].strip()
                
                # Find best matching speaker segment
                best_speaker = self._find_speaker_for_segment(
                    asr_start, asr_end, speaker_segments
                )
                
                # Create aligned segment
                aligned_segment = {
                    'start': asr_start,
                    'end': asr_end,
                    'duration': asr_end - asr_start,
                    'text': asr_text,
                    'speaker': best_speaker,
                    'words': asr_seg.get('words', [])
                }
                
                aligned_segments.append(aligned_segment)
            
            speakers = diarization_results['speakers']
            speaker_stats = self._calculate_speaker_text_stats(aligned_segments)
        
        # Combine all results
        combined_results = {
            'segments': aligned_segments,
            'speakers': speakers,
            'speaker_stats': speaker_stats,
            'asr_metadata': asr_results['metadata'],
            'diarization_metadata': diarization_results['metadata'],
            'pipeline_metadata': {
                'asr_engine': self.asr_engine_name,
                'asr_model': self.asr_model,
                'diarization_engines': self.diarization_engines,
                'total_segments': len(aligned_segments),
                'alignment_method': 'whisperx_integrated' if self.asr_engine_name == "whisperx" else 'overlap_based_enhanced',
                'preprocessing_enabled': self.enable_preprocessing,
                'language_detected': asr_results.get('language', 'Unknown'),
                'enhanced_features': self._get_asr_features()
            }
        }
        
        return combined_results
    
    def _find_speaker_for_segment(self, asr_start: float, asr_end: float, speaker_segments: List[Dict]) -> str:
        """Find the best matching speaker for an ASR segment"""
        
        best_overlap = 0
        best_speaker = "SPEAKER_UNKNOWN"
        
        for s_seg in speaker_segments:
            s_start, s_end = s_seg['start'], s_seg['end']
            
            # Calculate overlap between segments
            overlap_start = max(asr_start, s_start)
            overlap_end = min(asr_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio
            asr_duration = asr_end - asr_start
            overlap_ratio = overlap_duration / asr_duration if asr_duration > 0 else 0
            
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
        """Save enhanced multi-engine results with WhisperX information"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("  💾 Saving enhanced multi-engine results...")
        
        # 1. Complete JSON with all engine results
        json_path = output_dir / f"{base_name}_enhanced_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. Enhanced speaker transcript
        speaker_txt_path = output_dir / f"{base_name}_enhanced_transcript.txt"
        self._save_enhanced_transcript(results, speaker_txt_path)
        
        # 3. Enhanced Excel analysis
        excel_path = output_dir / f"{base_name}_enhanced_analysis.xlsx"
        self._save_enhanced_excel(results, excel_path)
        
        # 4. Basic preprocessing report (if preprocessing was used)
        if results['preprocessing'].get('applied') and 'summary' in results['preprocessing'].get('metrics', {}):
            preprocessing_report_path = output_dir / f"{base_name}_preprocessing_report.txt"
            self._save_preprocessing_report(results, preprocessing_report_path)
        
        # 5. WhisperX detailed report (if WhisperX was used)
        if self.asr_engine_name == "whisperx":
            whisperx_report_path = output_dir / f"{base_name}_whisperx_detailed.txt"
            self._save_whisperx_report(results, whisperx_report_path)
        
        print(f"     📁 Files saved to {output_dir}")
    
    def _save_enhanced_transcript(self, results: Dict, output_path: Path):
        """Save enhanced transcript with WhisperX and multi-engine information"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED MULTI-ENGINE SPEAKER-LABELED TRANSCRIPT\n")
            f.write("🚀 WHISPER + WHISPERX + WAV2VEC2 + MULTI-DIARIZATION\n")
            f.write("=" * 80 + "\n\n")
            
            # File and processing info
            asr_meta = results['asr_metadata']
            p_meta = results['pipeline_metadata']
            multi_engine = results.get('multi_engine', {})
            preprocessing = results.get('preprocessing', {})
            
            f.write(f"File: {asr_meta['file_name']}\n")
            f.write(f"Duration: {asr_meta['audio_duration_seconds']:.1f}s\n")
            f.write(f"Language: {p_meta.get('language_detected', asr_meta.get('language', 'Unknown'))}\n")
            f.write(f"Speakers: {len(results['speakers'])}\n")
            
            # Enhanced ASR Engine info
            f.write(f"ASR Engine: {multi_engine.get('asr_engine', 'unknown').upper()}\n")
            f.write(f"ASR Model: {multi_engine.get('asr_model', 'unknown')}\n")
            
            # Show enhanced features
            enhanced_features = multi_engine.get('asr_enhanced_features', [])
            if enhanced_features:
                f.write(f"Enhanced Features: {', '.join(enhanced_features)}\n")
            
            f.write(f"Diarization Engines: {', '.join(multi_engine.get('diarization_engines_used', []))}\n")
            f.write(f"Ensemble Mode: {'Yes' if multi_engine.get('ensemble_used', False) else 'No'}\n")
            f.write(f"WhisperX Built-in Used: {'Yes' if multi_engine.get('whisperx_builtin_used', False) else 'No'}\n")
            
            # Basic preprocessing info
            if preprocessing.get('applied'):
                metrics = preprocessing.get('metrics', {})
                f.write(f"Basic Preprocessing: Applied\n")
                f.write(f"Changes Made: {metrics.get('summary', 'Format standardization')}\n")
                f.write(f"Processing Effective: {'Yes' if metrics.get('processing_effective') else 'No'}\n")
            else:
                f.write(f"Basic Preprocessing: Not applied\n")
            
            # Engine comparison summary
            comparison = multi_engine.get('engine_comparison', {})
            if 'speaker_agreement' in comparison:
                f.write(f"Engine Agreement: {comparison['speaker_agreement']:.1f}%\n")
            
            f.write(f"Total Processing Time: {p_meta.get('total_processing_time', 0):.1f}s\n")
            
            # Enhanced performance metrics
            if self.asr_engine_name == "whisperx":
                whisperx_meta = asr_meta
                f.write(f"Forced Alignment Applied: {'Yes' if whisperx_meta.get('forced_alignment_applied') else 'No'}\n")
                f.write(f"Built-in Diarization: {'Yes' if whisperx_meta.get('diarization_applied') else 'No'}\n")
                f.write(f"Transcription Time: {whisperx_meta.get('transcription_time', 0):.1f}s\n")
                f.write(f"Alignment Time: {whisperx_meta.get('alignment_time', 0):.1f}s\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Speaker statistics
            f.write("SPEAKER SUMMARY:\n")
            for speaker, stats in results['speaker_stats'].items():
                f.write(f"{speaker}: {stats['total_duration']:.1f}s ({stats['duration_percentage']:.1f}%), ")
                f.write(f"{stats['total_words']} words ({stats['words_percentage']:.1f}%)\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            # Enhanced transcript with precise timestamps
            f.write("ENHANCED TRANSCRIPT:\n\n")
            for segment in results['segments']:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                text = segment['text']
                
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)
                
                # Show more precise timestamps for WhisperX
                if self.asr_engine_name == "whisperx":
                    f.write(f"[{int(start_min):02d}:{start_sec:06.3f} - {int(end_min):02d}:{end_sec:06.3f}] ")
                else:
                    f.write(f"[{int(start_min):02d}:{int(start_sec):02d} - {int(end_min):02d}:{int(end_sec):02d}] ")
                
                f.write(f"{speaker}: {text}\n\n")
    
    def _save_whisperx_report(self, results: Dict, output_path: Path):
        """Save detailed WhisperX-specific report"""
        
        if self.asr_engine_name != "whisperx":
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WHISPERX ENHANCED PROCESSING REPORT\n")
            f.write("🚀 Forced Alignment + Built-in Diarization\n")
            f.write("=" * 60 + "\n\n")
            
            asr_meta = results['asr_metadata']
            multi_engine = results['multi_engine']
            
            f.write(f"File: {asr_meta['file_name']}\n")
            f.write(f"WhisperX Model: {asr_meta['model_size']}\n")
            f.write(f"Compute Type: {asr_meta['compute_type']}\n")
            f.write(f"Device: {asr_meta['device']}\n")
            f.write(f"Batch Size: {asr_meta['batch_size']}\n\n")
            
            f.write("WHISPERX ENHANCED FEATURES:\n")
            f.write(f"Forced Alignment: {'✅' if asr_meta.get('forced_alignment_applied') else '❌'}\n")
            f.write(f"Built-in Diarization: {'✅' if asr_meta.get('diarization_applied') else '❌'}\n")
            f.write(f"Precise Timestamps: {'✅' if asr_meta.get('forced_alignment_applied') else '❌'}\n")
            f.write(f"Integrated Processing: {'✅' if multi_engine.get('whisperx_builtin_used') else '❌'}\n\n")
            
            f.write("PROCESSING PERFORMANCE:\n")
            f.write(f"Total Processing: {asr_meta.get('processing_time_seconds', 0):.1f}s\n")
            f.write(f"Transcription Phase: {asr_meta.get('transcription_time', 0):.1f}s\n")
            f.write(f"Alignment Phase: {asr_meta.get('alignment_time', 0):.1f}s\n")
            f.write(f"Diarization Phase: {asr_meta.get('diarization_time', 0):.1f}s\n")
            f.write(f"Speed Ratio: {asr_meta.get('speed_ratio', 0):.1f}x real-time\n\n")
            
            f.write("QUALITY ENHANCEMENTS:\n")
            f.write("• Forced alignment provides word-level precision\n")
            f.write("• Character-level alignment for maximum accuracy\n")
            f.write("• Integrated speaker diarization eliminates alignment errors\n")
            f.write("• Optimized batch processing for speed\n")
            f.write("• Multi-language alignment model support\n\n")
            
            if multi_engine.get('whisperx_builtin_used'):
                f.write("INTEGRATED DIARIZATION RESULTS:\n")
                speakers = results.get('speakers', [])
                f.write(f"Speakers Detected: {len(speakers)}\n")
                for speaker in speakers:
                    stats = results['speaker_stats'].get(speaker, {})
                    f.write(f"{speaker}: {stats.get('duration_percentage', 0):.1f}% of audio\n")
                f.write("\n")
            
            f.write("ADVANTAGES OVER STANDARD WHISPER:\n")
            f.write("• 10-100x more precise word timestamps\n")
            f.write("• Integrated speaker diarization\n")
            f.write("• No alignment drift between transcription and speakers\n")
            f.write("• Better handling of overlapping speech\n")
            f.write("• Enhanced multilingual support\n")
    
    def _save_preprocessing_report(self, results: Dict, output_path: Path):
        """Save detailed preprocessing report"""
        
        preprocessing = results.get('preprocessing', {})
        metrics = preprocessing.get('metrics', {})
        
        if not metrics:
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("BASIC AUDIO PREPROCESSING REPORT\n")
            f.write("🔧 Audio Format Standardization\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"File: {results['asr_metadata']['file_name']}\n")
            f.write(f"ASR Engine: {results['multi_engine'].get('asr_engine', 'unknown').upper()}\n")
            f.write(f"ASR Model: {results['multi_engine'].get('asr_model', 'unknown')}\n")
            f.write(f"Processing Type: {preprocessing.get('type', 'unknown')}\n")
            f.write(f"Purpose: {preprocessing.get('purpose', 'Format standardization')}\n\n")
            
            f.write("PROCESSING RESULTS:\n")
            f.write(f"Summary: {metrics.get('summary', 'N/A')}\n")
            f.write(f"Duration: {metrics.get('duration', 0):.1f}s\n")
            f.write(f"Sample Rate Changed: {'Yes' if metrics.get('sample_rate_changed') else 'No'}\n")
            f.write(f"Peak Change: {metrics.get('peak_change', 0):+.3f}\n")
            f.write(f"RMS Change: {metrics.get('rms_change', 0):+.3f}\n")
            f.write(f"DC Offset Change: {metrics.get('dc_offset_change', 0):+.4f}\n")
            f.write(f"Processing Effective: {'Yes' if metrics.get('processing_effective') else 'No'}\n\n")
            
            if 'original' in metrics and 'processed' in metrics:
                original = metrics['original']
                processed = metrics['processed']
                
                f.write("BEFORE vs AFTER:\n")
                f.write(f"Sample Rate: {original['sample_rate']}Hz → {processed['sample_rate']}Hz\n")
                f.write(f"Peak Level: {original['peak']:.3f} → {processed['peak']:.3f}\n")
                f.write(f"RMS Level: {original['rms']:.3f} → {processed['rms']:.3f}\n")
                f.write(f"DC Offset: {original['dc_offset']:.4f} → {processed['dc_offset']:.4f}\n")
                f.write(f"Clipping: {original['clipping_percentage']:.1f}% → {processed['clipping_percentage']:.1f}%\n\n")
            
            f.write("ENHANCED ENGINE COMPATIBILITY:\n")
            f.write("✅ Whisper (OpenAI)\n")
            f.write("✅ WhisperX (Enhanced with forced alignment)\n")
            f.write("✅ Wav2Vec2 (Facebook)\n")
            f.write("✅ PyAnnote (speaker diarization)\n")
            f.write("✅ SpeechBrain (speaker embeddings)\n")
            f.write("✅ NeMo (NVIDIA speech toolkit)\n")
            f.write("✅ VBX (clustering-based diarization)\n\n")
            
            if metrics.get('processing_effective'):
                f.write("RESULT: ✅ Audio successfully standardized for enhanced speech processing\n")
            else:
                f.write("RESULT: ℹ️  Audio was already in optimal format\n")
    
    def _save_enhanced_excel(self, results: Dict, output_path: Path):
        """Save enhanced Excel analysis with WhisperX information"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Enhanced Summary
            summary_data = [
                ['File Name', results['asr_metadata']['file_name']],
                ['Duration (seconds)', results['asr_metadata']['audio_duration_seconds']],
                ['Language', results['pipeline_metadata'].get('language_detected', results['asr_metadata'].get('language', 'Unknown'))],
                ['Number of Speakers', len(results['speakers'])],
                ['Total Segments', len(results['segments'])],
                ['', ''],
                ['ENHANCED ASR ENGINE', ''],
                ['ASR Engine', results['multi_engine'].get('asr_engine', 'unknown').upper()],
                ['ASR Model', results['multi_engine'].get('asr_model', 'unknown')],
                ['ASR Processing Time (s)', results['asr_metadata'].get('processing_time_seconds', 0)],
                ['ASR Speed Ratio', f"{results['asr_metadata'].get('speed_ratio', 0):.1f}x"],
                ['Enhanced Features', ', '.join(results['multi_engine'].get('asr_enhanced_features', []))],
                ['', ''],
            ]
            
            # Add WhisperX specific metrics
            if self.asr_engine_name == "whisperx":
                asr_meta = results['asr_metadata']
                summary_data.extend([
                    ['WHISPERX ENHANCED FEATURES', ''],
                    ['Forced Alignment Applied', 'Yes' if asr_meta.get('forced_alignment_applied') else 'No'],
                    ['Built-in Diarization', 'Yes' if asr_meta.get('diarization_applied') else 'No'],
                    ['Compute Type', asr_meta.get('compute_type', 'N/A')],
                    ['Batch Size', asr_meta.get('batch_size', 'N/A')],
                    ['Transcription Time (s)', asr_meta.get('transcription_time', 0)],
                    ['Alignment Time (s)', asr_meta.get('alignment_time', 0)],
                    ['Diarization Time (s)', asr_meta.get('diarization_time', 0)],
                    ['', ''],
                ])
            
            summary_data.extend([
                ['DIARIZATION ANALYSIS', ''],
                ['Engines Used', ', '.join(results['multi_engine'].get('diarization_engines_used', []))],
                ['Engines Failed', ', '.join(results['multi_engine'].get('diarization_engines_failed', [])) or 'None'],
                ['Ensemble Used', 'Yes' if results['multi_engine'].get('ensemble_used', False) else 'No'],
                ['WhisperX Built-in Used', 'Yes' if results['multi_engine'].get('whisperx_builtin_used', False) else 'No'],
                ['Total Processing Time (s)', results['pipeline_metadata'].get('total_processing_time', 0)],
                ['', ''],
                ['BASIC PREPROCESSING', ''],
                ['Preprocessing Type', results['preprocessing'].get('type', 'N/A')],
                ['Preprocessing Applied', 'Yes' if results['preprocessing'].get('applied') else 'No'],
                ['Purpose', results['preprocessing'].get('purpose', 'N/A')],
            ])
            
            # Add preprocessing metrics
            metrics = results['preprocessing'].get('metrics', {})
            if metrics and results['preprocessing'].get('applied'):
                summary_data.extend([
                    ['Changes Made', metrics.get('summary', 'N/A')],
                    ['Sample Rate Changed', 'Yes' if metrics.get('sample_rate_changed') else 'No'],
                    ['Peak Change', metrics.get('peak_change', 'N/A')],
                    ['RMS Change', metrics.get('rms_change', 'N/A')],
                    ['DC Offset Change', metrics.get('dc_offset_change', 'N/A')],
                    ['Processing Effective', 'Yes' if metrics.get('processing_effective') else 'No'],
                ])
            
            # Add engine comparison
            comparison = results['multi_engine'].get('engine_comparison', {})
            if comparison:
                summary_data.extend([
                    ['', ''],
                    ['ENGINE COMPARISON', ''],
                    ['Speaker Agreement (%)', f"{comparison.get('speaker_agreement', 0):.1f}"],
                    ['Success Rate (%)', f"{comparison.get('success_rate', 0):.1f}"],
                    ['Consensus Speakers', comparison.get('consensus_speakers', 'N/A')]
                ])
            
            df_summary = pd.DataFrame(summary_data, columns=['Property', 'Value'])
            df_summary.to_excel(writer, sheet_name='Enhanced_Summary', index=False)
            
            # Sheet 2: Engine Comparison Details (Enhanced)
            engine_results = results['multi_engine'].get('engine_results', {})
            engine_data = []
            
            for engine_name, result in engine_results.items():
                if 'error' in result:
                    engine_data.append({
                        'Engine': engine_name,
                        'Status': 'FAILED',
                        'Error': result['error'],
                        'Speakers': 'N/A',
                        'Segments': 'N/A',
                        'Duration': 'N/A',
                        'Processing_Time': 'N/A',
                        'Features': 'N/A'
                    })
                else:
                    features = ', '.join(comparison.get('features', {}).get(engine_name, []))
                    engine_data.append({
                        'Engine': engine_name,
                        'Status': 'SUCCESS',
                        'Error': '',
                        'Speakers': result['num_speakers'],
                        'Segments': len(result['segments']),
                        'Duration': result['total_duration'],
                        'Processing_Time': result.get('processing_time', 0),
                        'Features': features
                    })
            
            df_engines = pd.DataFrame(engine_data)
            df_engines.to_excel(writer, sheet_name='Engine_Comparison', index=False)
            
            # Sheet 3: Speaker Statistics
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
            
            # Sheet 4: Detailed Segments (Enhanced with precise timestamps)
            segments_data = []
            for i, segment in enumerate(results['segments'], 1):
                segments_data.append({
                    'Segment': i,
                    'Start_Time': segment['start'],
                    'End_Time': segment['end'],
                    'Duration': segment['duration'],
                    'Speaker': segment['speaker'],
                    'Word_Count': len(segment['text'].split()),
                    'Text': segment['text'],
                    'Has_Word_Timestamps': len(segment.get('words', [])) > 0
                })
            df_segments = pd.DataFrame(segments_data)
            df_segments.to_excel(writer, sheet_name='Enhanced_Segments', index=False)
    
    def print_enhanced_summary(self, results: Dict):
        """Print enhanced summary with WhisperX and multi-engine information"""
        
        duration = results['asr_metadata']['audio_duration_seconds']
        language = results['pipeline_metadata'].get('language_detected', results['asr_metadata'].get('language', 'Unknown'))
        num_speakers = len(results['speakers'])
        num_segments = len(results['segments'])
        multi_engine = results['multi_engine']
        preprocessing = results.get('preprocessing', {})
        
        print(f"✅ Enhanced Multi-Engine Processing Complete!")
        print(f"   📄 Duration: {duration:.1f}s")
        print(f"   🗣️  Language: {language}")
        print(f"   👥 Speakers: {num_speakers}")
        print(f"   📝 Segments: {num_segments}")
        
        # Enhanced ASR Engine info
        asr_features = multi_engine.get('asr_enhanced_features', [])
        print(f"   🎯 ASR Engine: {multi_engine.get('asr_engine', 'unknown').upper()} ({multi_engine.get('asr_model', 'unknown')})")
        print(f"   🚀 Features: {', '.join(asr_features)}")
        
        # WhisperX specific information
        if self.asr_engine_name == "whisperx":
            asr_meta = results['asr_metadata']
            print(f"   ⚡ Forced Alignment: {'✅' if asr_meta.get('forced_alignment_applied') else '❌'}")
            print(f"   🎙️  Built-in Diarization: {'✅' if asr_meta.get('diarization_applied') else '❌'}")
            print(f"   ⏱️  Transcription: {asr_meta.get('transcription_time', 0):.1f}s")
            print(f"   ⏱️  Alignment: {asr_meta.get('alignment_time', 0):.1f}s")
        
        print(f"   🔧 Diarization: {', '.join(multi_engine.get('diarization_engines_used', []))}")
        print(f"   🤖 Ensemble: {'✅' if multi_engine.get('ensemble_used', False) else '❌'}")
        print(f"   🎯 WhisperX Built-in: {'✅' if multi_engine.get('whisperx_builtin_used', False) else '❌'}")
        
        # Basic preprocessing info
        if preprocessing.get('applied'):
            metrics = preprocessing.get('metrics', {})
            print(f"   🔧 Basic Preprocessing: ✅")
            print(f"   📈 Changes: {metrics.get('summary', 'Format standardization')}")
            if metrics.get('processing_effective'):
                print(f"   ✅ Audio format standardized for enhanced engines")
            else:
                print(f"   ℹ️  Audio was already well-formatted")
        else:
            print(f"   🔧 Basic Preprocessing: ❌")
        
        # Engine comparison
        comparison = multi_engine.get('engine_comparison', {})
        if 'speaker_agreement' in comparison:
            print(f"   🎯 Agreement: {comparison['speaker_agreement']:.1f}%")
        
        if multi_engine.get('diarization_engines_failed'):
            print(f"   ⚠️  Failed: {', '.join(multi_engine['diarization_engines_failed'])}")
        
        print(f"   ⏱️  Total Time: {results['pipeline_metadata'].get('total_processing_time', 0):.1f}s")
        
        # Show top speakers
        sorted_speakers = sorted(
            results['speaker_stats'].items(), 
            key=lambda x: x[1]['total_duration'], 
            reverse=True
        )
        
        print(f"   🎙️  Speaker breakdown:")
        for speaker, stats in sorted_speakers[:3]:  # Show top 3
            duration_pct = stats['duration_percentage']
            word_count = stats['total_words']
            print(f"      {speaker}: {duration_pct:.1f}%, {word_count} words")
        
        if len(sorted_speakers) > 3:
            print(f"      ... and {len(sorted_speakers) - 3} more speakers")
    
    def cleanup(self):
        """Clean up resources"""
        if self.preprocessor:
            self.preprocessor.cleanup()


# Enhanced utility functions
def find_audio_files(directory: str = ".") -> List[Path]:
    """Find all audio files in the specified directory"""
    audio_extensions = ["*.mp3", "*.wav", "*.mp4", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    audio_files = []
    
    for extension in audio_extensions:
        files = list(Path(directory).glob(extension))
        files.extend(list(Path(directory).glob(extension.upper())))
        audio_files.extend(files)
    
    unique_files = []
    seen = set()
    for file in audio_files:
        if file.resolve() not in seen:
            seen.add(file.resolve())
            unique_files.append(file)
    
    return sorted(unique_files)


def select_audio_file() -> Optional[Path]:
    """Let user select an audio file"""
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


def select_enhanced_asr_engine() -> Tuple[str, str]:
    """Let user select ASR engine and model with WhisperX support"""
    
    print("\n🎯 Select Enhanced ASR Engine:")
    print("   1. Whisper (OpenAI) - Multilingual, robust")
    print("   2. WhisperX (Enhanced) - Forced alignment + built-in diarization (BEST)")
    print("   3. Wav2Vec2 (Facebook) - German specialist")
    
    while True:
        try:
            choice = input("\nSelect ASR engine (1-3) or Enter for WhisperX: ").strip()
            
            if not choice or choice == "2":
                return "whisperx", "large-v3"
            elif choice == "1":
                return "whisper", "large-v3"
            elif choice == "3":
                print("\n🇩🇪 Select Wav2Vec2 model:")
                print("   1. German - Optimized for German (RECOMMENDED)")
                print("   2. English - English specialization")
                print("   3. Multilingual - Multiple languages")
                
                model_choice = input("\nSelect model (1-3) or Enter for German: ").strip()
                
                if not model_choice or model_choice == "1":
                    return "wav2vec2", "german"
                elif model_choice == "2":
                    return "wav2vec2", "english"
                else:
                    return "wav2vec2", "multilingual"
            else:
                print("Please enter 1-3")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return "whisperx", "large-v3"


def select_enhanced_diarization_engines(asr_engine: str) -> List[str]:
    """Let user select diarization engine(s) with WhisperX awareness"""
    
    engines = EnhancedMultiEngineSpeechPipeline.SUPPORTED_DIARIZATION_ENGINES
    
    print("\n🔧 Select Diarization Engine(s):")
    for i, (key, name) in enumerate(engines.items(), 1):
        if key == "whisperx_builtin" and asr_engine != "whisperx":
            print(f"   {i}. {name} (Requires WhisperX ASR)")
        else:
            print(f"   {i}. {name}")
    
    print("\n   You can select multiple engines (e.g., '1,3,4') or single engine")
    if asr_engine == "whisperx":
        print("   💡 WhisperX built-in diarization is recommended for best integration")
    
    while True:
        try:
            if asr_engine == "whisperx":
                default_choice = "whisperx_builtin"
                choice = input("\nSelect engine(s) (1-6) or Enter for WhisperX built-in: ").strip()
            else:
                default_choice = "pyannote"
                choice = input("\nSelect engine(s) (1-6) or Enter for PyAnnote: ").strip()
            
            if not choice:
                return [default_choice]
            
            # Parse selection
            if ',' in choice:
                # Multiple engines
                indices = [int(x.strip()) for x in choice.split(',')]
                selected_engines = []
                engine_keys = list(engines.keys())
                
                for idx in indices:
                    if 1 <= idx <= len(engine_keys):
                        engine_key = engine_keys[idx-1]
                        if engine_key == "whisperx_builtin" and asr_engine != "whisperx":
                            print(f"   ⚠️  Skipping WhisperX built-in (requires WhisperX ASR)")
                        else:
                            selected_engines.append(engine_key)
                    else:
                        print(f"Invalid selection: {idx}")
                        continue
                
                if selected_engines:
                    return selected_engines
                else:
                    print("No valid engines selected, using default")
                    return [default_choice]
                    
            else:
                # Single engine
                idx = int(choice)
                engine_keys = list(engines.keys())
                if 1 <= idx <= len(engine_keys):
                    engine_key = engine_keys[idx-1]
                    if engine_key == "whisperx_builtin" and asr_engine != "whisperx":
                        print(f"   ⚠️  WhisperX built-in requires WhisperX ASR, using PyAnnote")
                        return ["pyannote"]
                    else:
                        return [engine_key]
                else:
                    print(f"Please enter 1-{len(engine_keys)}")
                    
        except ValueError:
            print("Please enter valid number(s)")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return [default_choice]


def ask_about_preprocessing() -> bool:
    """Ask user if they want to apply basic preprocessing"""
    
    print("\n🔧 Audio Preprocessing Options:")
    print("   Basic preprocessing will:")
    print("   • Convert to 16kHz mono")
    print("   • Remove DC offset")
    print("   • Normalize audio levels")
    print("   • Ensure compatibility with all speech engines")
    print()
    print("   Recommended: YES (optimal for WhisperX and all engines)")
    
    while True:
        try:
            choice = input("\nApply basic preprocessing? (y/n, default=y): ").strip().lower()
            
            if not choice or choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return True  # Default to yes


def main():
    """Main pipeline execution with enhanced WhisperX support"""
    
    try:
        print("🎤 Enhanced Multi-Engine Speech Diarization Pipeline")
        print("🚀 WITH WHISPERX FORCED ALIGNMENT + BUILT-IN DIARIZATION")
        print("=" * 70)
        
        # Test WhisperX installation
        try:
            import whisperx
            print("✅ WhisperX available for enhanced processing")
        except ImportError:
            print("⚠️  WhisperX not installed - install with: pip install whisperx")
            print("   Falling back to standard engines")
        
        # Select audio file
        audio_file = select_audio_file()
        if not audio_file:
            return
        
        # Select enhanced ASR engine and model
        asr_engine, asr_model = select_enhanced_asr_engine()
        
        # Select diarization engine(s) with WhisperX awareness
        diarization_engines = select_enhanced_diarization_engines(asr_engine)
        
        # Ask about preprocessing
        apply_preprocessing = ask_about_preprocessing()
        
        print()  # Empty line for clarity
        
        # Initialize enhanced multi-engine pipeline
        pipeline = EnhancedMultiEngineSpeechPipeline(
            asr_engine=asr_engine,
            asr_model=asr_model,
            diarization_engines=diarization_engines,
            device="auto",
            enable_preprocessing=True,  # Always enable, but let user choose whether to apply
            whisperx_enable_diarization=True  # Enable WhisperX built-in diarization
        )
        
        # Process audio with enhanced features
        results = pipeline.process_audio(
            audio_path=audio_file,
            language=None,  # Auto-detect
            num_speakers=None,
            min_speakers=1,
            max_speakers=10,
            apply_preprocessing=apply_preprocessing,
            enable_ensemble=len(diarization_engines) > 1 or "ensemble" in diarization_engines
        )
        
        # Save enhanced results
        preprocessing_suffix = "_with_preprocessing" if apply_preprocessing else "_no_preprocessing"
        output_dir = f"enhanced_{asr_engine}_output{preprocessing_suffix}"
        base_name = audio_file.stem
        pipeline.save_results(results, output_dir, base_name)
        
        # Print enhanced summary
        pipeline.print_enhanced_summary(results)
        
        # Show preprocessing summary if applied
        if apply_preprocessing and results['preprocessing'].get('applied'):
            metrics = results['preprocessing']['metrics']
            print(f"\n🔧 Preprocessing Summary:")
            print(f"   Changes: {metrics.get('summary', 'Format standardization')}")
            if metrics.get('processing_effective'):
                print(f"   ✅ Audio format optimized for enhanced engines")
            else:
                print(f"   ℹ️  Audio was already well-formatted")
        elif apply_preprocessing:
            print(f"\n🔧 Preprocessing: Requested but not applied (error occurred)")
        else:
            print(f"\n🔧 Preprocessing: Skipped (using original audio)")
        
        # Show enhanced ASR-specific summary
        asr_meta = results['asr_metadata']
        print(f"\n🎯 Enhanced ASR Summary:")
        print(f"   Engine: {asr_engine.upper()}")
        print(f"   Model: {asr_model}")
        print(f"   Language: {results['pipeline_metadata'].get('language_detected', 'Unknown')}")
        print(f"   Processing: {asr_meta.get('processing_time_seconds', 0):.1f}s")
        print(f"   Speed: {asr_meta.get('speed_ratio', 0):.1f}x real-time")
        
        # WhisperX specific summary
        if asr_engine == 'whisperx':
            print(f"   🚀 Enhanced Features:")
            if asr_meta.get('forced_alignment_applied'):
                print(f"      ⚡ Forced alignment: Precise word timestamps")
            if asr_meta.get('diarization_applied'):
                print(f"      👥 Built-in diarization: Integrated speaker detection")
            print(f"      🎯 Transcription: {asr_meta.get('transcription_time', 0):.1f}s")
            print(f"      🎯 Alignment: {asr_meta.get('alignment_time', 0):.1f}s")
            print(f"      🎯 Diarization: {asr_meta.get('diarization_time', 0):.1f}s")
        elif asr_engine == 'wav2vec2' and results['pipeline_metadata'].get('language_detected') == 'de':
            print(f"   🇩🇪 German language optimization: ACTIVE")
            print(f"   ✅ Wav2Vec2 provides superior German accuracy")
        
        print(f"\n📁 Enhanced files saved in '{output_dir}/' directory")
        print("🎉 Enhanced multi-engine processing complete!")
        
        # Cleanup
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        print(f"\n👋 Cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()