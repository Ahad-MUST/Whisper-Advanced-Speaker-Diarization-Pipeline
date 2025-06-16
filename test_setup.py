# test_setup.py - Complete Setup Verification Script

import sys
import warnings
warnings.filterwarnings("ignore")

def test_basic_imports():
    """Test if all basic packages can be imported"""
    print("🔍 Testing Basic Imports...")
    
    tests = [
        ("torch", "PyTorch"),
        ("librosa", "Librosa"), 
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("tqdm", "Progress bars"),
        ("json", "JSON support"),
        ("pathlib", "Path handling")
    ]
    
    failed = []
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(module)
    
    return len(failed) == 0

def test_pytorch_cuda():
    """Test PyTorch and CUDA setup"""
    print("\n🔍 Testing PyTorch & CUDA...")
    
    try:
        import torch
        print(f"  ✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ CUDA available: {torch.version.cuda}")
            print(f"  ✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print(f"  ⚠️  CUDA not available - will use CPU")
            print(f"     This will work but be slower")
            return True
            
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False

def test_whisper():
    """Test Whisper installation"""
    print("\n🔍 Testing Whisper...")
    
    try:
        import whisper
        print(f"  ✅ Whisper imported successfully")
        
        # Test model loading
        print(f"  🔄 Testing tiny model load...")
        model = whisper.load_model("tiny")
        print(f"  ✅ Whisper model loaded successfully")
        return True
        
    except ImportError as e:
        print(f"  ❌ Whisper import failed: {e}")
        print(f"     Run: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"  ❌ Whisper model load failed: {e}")
        return False

def test_pyannote_imports():
    """Test PyAnnote package imports"""
    print("\n🔍 Testing PyAnnote Imports...")
    
    try:
        from pyannote.audio import Pipeline
        print(f"  ✅ PyAnnote audio imported")
        
        import transformers
        print(f"  ✅ Transformers available: {transformers.__version__}")
        
        from huggingface_hub import whoami
        print(f"  ✅ HuggingFace hub available")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ PyAnnote import failed: {e}")
        print(f"     Run: pip install pyannote.audio transformers huggingface-hub")
        return False

def test_huggingface_auth():
    """Test HuggingFace authentication"""
    print("\n🔍 Testing HuggingFace Authentication...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"  ✅ Authenticated as: {user_info['name']}")
        return True
        
    except Exception as e:
        print(f"  ❌ Authentication failed: {e}")
        print(f"     Solutions:")
        print(f"     1. Create account: https://huggingface.co/join")
        print(f"     2. Get token: https://huggingface.co/settings/tokens")
        print(f"     3. Run: huggingface-cli login")
        return False

def test_pyannote_model_access():
    """Test PyAnnote model access"""
    print("\n🔍 Testing PyAnnote Model Access...")
    
    try:
        from pyannote.audio import Pipeline
        
        print(f"  🔄 Attempting to load diarization model...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True
        )
        print(f"  ✅ PyAnnote model access successful")
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        
        if "not found" in error_str or "does not exist" in error_str:
            print(f"  ❌ Model access failed: {e}")
            print(f"     Solution: Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1")
        elif "authentication" in error_str or "token" in error_str:
            print(f"  ❌ Authentication issue: {e}")
            print(f"     Solution: Run 'huggingface-cli login' with valid token")
        else:
            print(f"  ❌ Unexpected error: {e}")
        
        return False

def test_audio_files():
    """Check for audio files to test with"""
    print("\n🔍 Looking for Audio Files...")
    
    from pathlib import Path
    
    audio_extensions = ['.mp3', '.wav', '.mp4', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path('.').glob(f'*{ext}'))
        audio_files.extend(Path('.').glob(f'*{ext.upper()}'))
    
    if audio_files:
        print(f"  ✅ Found {len(audio_files)} audio file(s):")
        for file in audio_files[:3]:  # Show first 3
            size_mb = file.stat().st_size / 1e6
            print(f"     - {file.name} ({size_mb:.1f} MB)")
        if len(audio_files) > 3:
            print(f"     ... and {len(audio_files) - 3} more")
        return True
    else:
        print(f"  ⚠️  No audio files found for testing")
        print(f"     Add some audio files (MP3, WAV, etc.) to test the pipeline")
        return False

def test_pipeline_files():
    """Check if all pipeline files are present"""
    print("\n🔍 Checking Pipeline Files...")
    
    from pathlib import Path
    
    required_files = [
        "whisper_engine.py",
        "pyannote_engine.py", 
        "pipeline_integration.py",
        "requirements.txt"
    ]
    
    missing = []
    
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing!")
            missing.append(file)
    
    return len(missing) == 0

def run_complete_test():
    """Run all tests and provide summary"""
    
    print("🧪 Complete Pipeline Setup Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("Whisper", test_whisper),
        ("PyAnnote Imports", test_pyannote_imports),
        ("HuggingFace Auth", test_huggingface_auth),
        ("PyAnnote Model Access", test_pyannote_model_access),
        ("Pipeline Files", test_pipeline_files),
        ("Audio Files", test_audio_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ Test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your setup is ready for the complete pipeline")
        print("🚀 Run: python pipeline_integration.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("❌ Please fix the issues above before running the pipeline")
    
    return passed == total

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)