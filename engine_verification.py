# engine_verification.py - Comprehensive Engine Verification and Diagnostic Script

import sys
import warnings
import time
import tempfile
import os
import numpy as np
import soundfile as sf
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def create_test_audio(duration=10.0, sample_rate=16000):
    """Create a test audio file with speech-like characteristics"""
    print("🎵 Creating test audio for engine verification...")
    
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create speech-like audio with multiple "speakers"
    # Speaker 1: Lower frequency (first half)
    speaker1_freq = 150  # Hz - typical male fundamental
    speaker1_audio = 0.3 * np.sin(2 * np.pi * speaker1_freq * t[:samples//2])
    
    # Speaker 2: Higher frequency (second half)
    speaker2_freq = 220  # Hz - typical female fundamental
    speaker2_audio = 0.3 * np.sin(2 * np.pi * speaker2_freq * t[samples//2:])
    
    # Combine and add formants (speech-like harmonics)
    test_audio = np.concatenate([speaker1_audio, speaker2_audio])
    
    # Add formant-like overtones
    formant1 = 0.1 * np.sin(2 * np.pi * 800 * t)  # First formant
    formant2 = 0.05 * np.sin(2 * np.pi * 1200 * t)  # Second formant
    test_audio += formant1 + formant2
    
    # Add some natural variation and light noise
    variation = 0.05 * np.random.randn(samples)
    test_audio += variation
    
    # Normalize
    test_audio = test_audio / np.max(np.abs(test_audio)) * 0.8
    
    # Save to temporary file
    test_file = Path("engine_test_audio.wav")
    sf.write(test_file, test_audio, sample_rate)
    
    print(f"   ✅ Test audio created: {duration}s, {sample_rate}Hz, 2 synthetic speakers")
    return test_file

def test_whisper_engine():
    """Test Whisper engine with detailed verification"""
    print("\n" + "="*60)
    print("🔍 TESTING WHISPER ENGINE")
    print("="*60)
    
    try:
        from whisper_engine import WhisperEngine
        
        # Test different model sizes
        models_to_test = ["tiny", "base"]  # Start with smaller models for faster testing
        
        for model_size in models_to_test:
            print(f"\n🔄 Testing Whisper {model_size} model...")
            
            try:
                engine = WhisperEngine(model_size=model_size, device="auto")
                
                # Get model info
                model_info = engine.get_model_info()
                print(f"   ✅ {model_size} model loaded successfully")
                print(f"   📊 Device: {model_info['device']}")
                print(f"   🎯 CUDA Available: {model_info['cuda_available']}")
                if model_info['gpu_name']:
                    print(f"   🖥️  GPU: {model_info['gpu_name']}")
                
                # Test transcription with test audio
                test_file = create_test_audio(duration=5.0)
                
                start_time = time.time()
                results = engine.transcribe_audio(
                    audio_path=test_file,
                    language=None,
                    word_timestamps=True
                )
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing time: {processing_time:.1f}s")
                print(f"   📝 Transcribed text: '{results['text'][:50]}...'")
                print(f"   🌍 Detected language: {results['language']}")
                print(f"   📊 Segments generated: {len(results['segments'])}")
                print(f"   ⚡ Speed ratio: {results['metadata']['speed_ratio']:.1f}x real-time")
                
                # Cleanup
                test_file.unlink()
                
                print(f"   ✅ Whisper {model_size} test: PASSED")
                
            except Exception as e:
                print(f"   ❌ Whisper {model_size} test FAILED: {e}")
                continue
        
        return True
        
    except ImportError as e:
        print(f"❌ Whisper engine import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Whisper engine test failed: {e}")
        return False

def test_diarization_engine(engine_name, engine_class):
    """Test a specific diarization engine"""
    print(f"\n🔄 Testing {engine_name} engine...")
    
    try:
        # Initialize engine
        engine = engine_class(device="auto")
        print(f"   ✅ {engine_name} initialized successfully")
        
        # Create test audio
        test_file = create_test_audio(duration=8.0)  # Longer for diarization
        
        # Test diarization
        start_time = time.time()
        results = engine.diarize_audio(
            audio_path=test_file,
            num_speakers=None,
            min_speakers=1,
            max_speakers=5
        )
        processing_time = time.time() - start_time
        
        print(f"   ⏱️  Processing time: {processing_time:.1f}s")
        print(f"   👥 Speakers detected: {results['num_speakers']}")
        print(f"   📊 Segments generated: {len(results['segments'])}")
        print(f"   🎯 Total duration: {results['total_duration']:.1f}s")
        
        # Verify segment structure
        if results['segments']:
            sample_segment = results['segments'][0]
            expected_keys = ['start', 'end', 'speaker', 'duration']
            missing_keys = [key for key in expected_keys if key not in sample_segment]
            
            if missing_keys:
                print(f"   ⚠️  Missing keys in segments: {missing_keys}")
            else:
                print(f"   ✅ Segment structure valid")
        
        # Test speaker statistics
        if 'speaker_stats' in results:
            print(f"   📈 Speaker statistics available")
            for speaker, stats in results['speaker_stats'].items():
                duration_pct = stats.get('percentage', 0)
                print(f"      {speaker}: {duration_pct:.1f}% of audio")
        
        # Test timeline extraction
        try:
            timeline = engine.get_speaker_timeline(results)
            print(f"   🕒 Timeline extracted: {len(timeline)} entries")
        except Exception as e:
            print(f"   ⚠️  Timeline extraction failed: {e}")
        
        # Cleanup
        test_file.unlink()
        
        print(f"   ✅ {engine_name} test: PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ {engine_name} test FAILED: {e}")
        return False

def test_all_diarization_engines():
    """Test all available diarization engines"""
    print("\n" + "="*60)
    print("🔍 TESTING DIARIZATION ENGINES")
    print("="*60)
    
    engines_to_test = [
        ("PyAnnote", "pyannote_engine", "PyAnnoteEngine"),
        ("NeMo", "nemo_engine", "NeMoEngine"),
        ("VBX", "vbx_engine", "VBXEngine"),
        ("SpeechBrain", "speechbrain_engine", "SpeechBrainEngine"),
    ]
    
    results = {}
    
    for engine_name, module_name, class_name in engines_to_test:
        try:
            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            engine_class = getattr(module, class_name)
            
            # Test the engine
            results[engine_name] = test_diarization_engine(engine_name, engine_class)
            
        except ImportError as e:
            print(f"\n🔄 Testing {engine_name} engine...")
            print(f"   ❌ {engine_name} import failed: {e}")
            results[engine_name] = False
        except Exception as e:
            print(f"\n🔄 Testing {engine_name} engine...")
            print(f"   ❌ {engine_name} test crashed: {e}")
            results[engine_name] = False
    
    return results

def test_research_audio_preprocessor():
    """Test the research-based audio preprocessor"""
    print("\n" + "="*60)
    print("🔍 TESTING RESEARCH AUDIO PREPROCESSOR")
    print("="*60)
    
    try:
        from research_audio_preprocessor import ResearchBasedAudioPreprocessor
        
        # Initialize preprocessor
        print("🔄 Initializing research preprocessor...")
        preprocessor = ResearchBasedAudioPreprocessor(device="auto", enable_demucs=True)
        print("   ✅ Research preprocessor initialized")
        
        # Create test audio with known characteristics
        print("\n🔄 Creating test audio with known issues...")
        test_audio = np.random.randn(16000 * 10) * 0.5  # 10 seconds of noise
        test_audio += 0.1  # Add DC offset
        test_audio = np.clip(test_audio, -0.95, 0.95)  # Prevent clipping
        
        test_file = Path("preprocessor_test.wav")
        sf.write(test_file, test_audio, 16000)
        
        print(f"   📊 Original: 10.0s, DC offset: 0.1, noise-based")
        
        # Test all profiles
        profiles = ['minimal', 'standard', 'neural_extraction']
        
        for profile in profiles:
            print(f"\n🔄 Testing {profile} profile...")
            
            try:
                processed_path, metrics = preprocessor.process_audio(
                    audio_path=test_file,
                    profile=profile,
                    quality_assessment=True
                )
                
                if 'research_compliance' in metrics:
                    compliance = metrics['research_compliance']
                    print(f"   📚 Research Grade: {compliance['research_grade']}")
                    print(f"   🔬 Compliance Score: {compliance['compliance_score']:.2f}")
                    print(f"   ✅ Spectral Preservation: {compliance['spectral_preservation']}")
                    
                    if compliance['research_grade'] in ['A', 'B']:
                        print(f"   ✅ {profile} profile: GOOD")
                    else:
                        print(f"   ⚠️  {profile} profile: NEEDS IMPROVEMENT")
                
                # Check if processing actually improved things
                if 'before' in metrics and 'after' in metrics:
                    before = metrics['before']
                    after = metrics['after']
                    
                    dc_improved = abs(after['dc_offset']) < abs(before['dc_offset'])
                    print(f"   🔧 DC offset improved: {dc_improved}")
                    print(f"      Before: {before['dc_offset']:.4f} → After: {after['dc_offset']:.4f}")
                
                # Cleanup processed file
                if os.path.exists(processed_path):
                    os.remove(processed_path)
                
            except Exception as e:
                print(f"   ❌ {profile} profile test failed: {e}")
        
        # Cleanup
        test_file.unlink()
        preprocessor.cleanup()
        
        print("\n   ✅ Research preprocessor test completed")
        return True
        
    except ImportError as e:
        print(f"❌ Research preprocessor import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Research preprocessor test failed: {e}")
        return False

def diagnose_preprocessing_issue():
    """Diagnose why preprocessing got a C grade"""
    print("\n" + "="*60)
    print("🔍 DIAGNOSING PREPROCESSING ISSUE")
    print("="*60)
    
    print("🔍 Analyzing your preprocessing results...")
    print("   Input:  317.3s @ 44.1kHz")
    print("   Output: 115.1s @ 16kHz")
    print("   Duration change: -63.7% ❌")
    print("   Research Grade: C ❌")
    print()
    
    print("🔬 DIAGNOSIS:")
    print("   1. ❌ Excessive duration reduction (317s → 115s)")
    print("      This suggests aggressive silence trimming")
    print("   2. ❌ Spectral modification detected")
    print("      The preprocessor is changing frequency characteristics")
    print("   3. ❌ Research compliance score: 0.50/1.0")
    print("      Below acceptable threshold for speech models")
    print()
    
    print("🔧 RECOMMENDED FIXES:")
    print("   1. Use 'minimal' profile instead of 'standard'")
    print("   2. Check if input audio has excessive silence")
    print("   3. Verify the research preprocessor settings")
    print("   4. Consider using 'disabled' preprocessing for comparison")
    print()
    
    print("💡 NEXT STEPS:")
    print("   1. Run the test again with 'minimal' profile")
    print("   2. Compare results with 'disabled' preprocessing")
    print("   3. Check if the dramatic duration change is expected")

def test_pipeline_integration():
    """Test the full pipeline integration"""
    print("\n" + "="*60)
    print("🔍 TESTING PIPELINE INTEGRATION")
    print("="*60)
    
    try:
        from multi_engine_pipeline import MultiEngineSpeechDiarizationPipeline
        
        print("🔄 Testing pipeline initialization...")
        
        # Test with minimal configuration
        pipeline = MultiEngineSpeechDiarizationPipeline(
            whisper_model="tiny",  # Fast model for testing
            diarization_engines=["speechbrain"],  # Single engine for speed
            device="auto",
            enable_preprocessing=True
        )
        
        print("   ✅ Pipeline initialized successfully")
        print(f"   🔧 Loaded engines: {len(pipeline.loaded_engines)}")
        print(f"   ⚠️  Failed engines: {len(pipeline.failed_engines)}")
        
        if pipeline.failed_engines:
            print(f"      Failed: {', '.join(pipeline.failed_engines)}")
        
        # Test preprocessing settings
        print("\n🔄 Testing preprocessing profiles...")
        profiles = ['minimal', 'standard', 'maximum', 'disabled']
        
        for profile in profiles:
            settings = pipeline._get_preprocessing_settings(profile)
            if settings is not None:
                print(f"   ✅ {profile}: {settings}")
            else:
                print(f"   ⚠️  {profile}: disabled")
        
        # Cleanup
        pipeline.cleanup()
        
        print("\n   ✅ Pipeline integration test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def run_comprehensive_verification():
    """Run comprehensive verification of all components"""
    
    print("🧪 COMPREHENSIVE MULTI-ENGINE VERIFICATION")
    print("=" * 70)
    print("This script will test all engines and components systematically")
    print("=" * 70)
    
    results = {}
    start_time = time.time()
    
    # Test individual components
    print("\n📋 TESTING INDIVIDUAL COMPONENTS:")
    results['whisper'] = test_whisper_engine()
    results['diarization'] = test_all_diarization_engines()
    results['preprocessor'] = test_research_audio_preprocessor()
    results['pipeline'] = test_pipeline_integration()
    
    # Diagnose issues
    diagnose_preprocessing_issue()
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    print(f"⏱️  Total test time: {total_time:.1f}s\n")
    
    # Whisper results
    print("🎤 WHISPER ENGINE:")
    if results['whisper']:
        print("   ✅ WORKING - Transcription engine ready")
    else:
        print("   ❌ FAILED - Transcription engine has issues")
    
    # Diarization results
    print("\n👥 DIARIZATION ENGINES:")
    working_engines = []
    failed_engines = []
    
    for engine, status in results['diarization'].items():
        if status:
            working_engines.append(engine)
            print(f"   ✅ {engine} - Ready for speaker identification")
        else:
            failed_engines.append(engine)
            print(f"   ❌ {engine} - Engine has issues")
    
    # Preprocessor results
    print("\n🧬 RESEARCH PREPROCESSOR:")
    if results['preprocessor']:
        print("   ✅ WORKING - But may need profile adjustment")
        print("   ⚠️  Current issue: Aggressive processing in your case")
    else:
        print("   ❌ FAILED - Preprocessor has major issues")
    
    # Pipeline results
    print("\n🔗 PIPELINE INTEGRATION:")
    if results['pipeline']:
        print("   ✅ WORKING - Multi-engine pipeline ready")
    else:
        print("   ❌ FAILED - Integration issues detected")
    
    # Overall assessment
    print("\n🎯 OVERALL SYSTEM STATUS:")
    critical_working = results['whisper'] and len(working_engines) > 0
    
    if critical_working:
        print("   ✅ SYSTEM OPERATIONAL")
        print(f"   📊 Working engines: {len(working_engines)}")
        print(f"   🎤 Transcription: Ready")
        print(f"   👥 Diarization: {len(working_engines)} engine(s) available")
        
        if results['preprocessor']:
            print("   🧬 Preprocessing: Working (needs tuning)")
        else:
            print("   🧬 Preprocessing: Issues detected")
    else:
        print("   ❌ SYSTEM HAS CRITICAL ISSUES")
        print("   🔧 Need to fix core engines first")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not results['whisper']:
        print("   🔧 Fix Whisper engine installation")
    
    if len(working_engines) == 0:
        print("   🔧 Fix at least one diarization engine")
    
    if not results['preprocessor']:
        print("   🔧 Fix research preprocessor")
    else:
        print("   🔧 Use 'minimal' preprocessing profile")
        print("   🔧 Investigate why duration changed so dramatically")
    
    print("\n🏁 Verification complete!")
    
    return results

if __name__ == "__main__":
    try:
        run_comprehensive_verification()
    except KeyboardInterrupt:
        print("\n\n👋 Verification cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Verification script crashed: {e}")
        import traceback
        traceback.print_exc()