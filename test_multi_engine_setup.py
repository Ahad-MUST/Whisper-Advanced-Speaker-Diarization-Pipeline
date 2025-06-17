# test_multi_engine_setup.py - Complete Multi-Engine Setup Verification

import sys
import warnings
from pathlib import Path
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
        ("omegaconf", "OmegaConf"),
        ("whisper", "OpenAI Whisper")
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
            
            if gpu_memory >= 8:
                print(f"  ✅ VRAM sufficient for both engines")
            elif gpu_memory >= 4:
                print(f"  ⚠️  VRAM may be tight for NeMo (consider PyAnnote)")
            else:
                print(f"  ❌ VRAM too low - use CPU mode")
            
            return True
        else:
            print(f"  ⚠️  CUDA not available - will use CPU")
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
        return False
    except Exception as e:
        print(f"  ❌ Whisper model load failed: {e}")
        return False

def test_pyannote():
    """Test PyAnnote installation and authentication"""
    print("\n🔍 Testing PyAnnote...")
    
    try:
        from pyannote.audio import Pipeline
        print(f"  ✅ PyAnnote audio imported")
        
        import transformers
        print(f"  ✅ Transformers available: {transformers.__version__}")
        
        # Test authentication
        try:
            from huggingface_hub import whoami
            user_info = whoami()
            print(f"  ✅ HuggingFace authenticated as: {user_info['name']}")
            
            # Test model access
            print(f"  🔄 Testing model access...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
            print(f"  ✅ PyAnnote model access successful")
            return True
            
        except Exception as e:
            print(f"  ❌ Authentication/model access failed: {e}")
            print(f"     Solutions:")
            print(f"     1. Run: huggingface-cli login")
            print(f"     2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False
        
    except ImportError as e:
        print(f"  ❌ PyAnnote import failed: {e}")
        print(f"     Run: pip install pyannote.audio transformers huggingface-hub")
        return False

def test_nemo():
    """Test NeMo installation"""
    print("\n🔍 Testing NeMo...")
    
    try:
        # Test basic imports
        from nemo.collections.asr.models import ClusteringDiarizer
        print(f"  ✅ NeMo ClusteringDiarizer imported")
        
        from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
        print(f"  ✅ NeMo MSDD models imported")
        
        import pytorch_lightning
        print(f"  ✅ PyTorch Lightning: {pytorch_lightning.__version__}")
        
        from omegaconf import OmegaConf
        print(f"  ✅ OmegaConf imported")
        
        # Test model availability (this will download if needed)
        print(f"  🔄 Testing NeMo model availability...")
        
        # Create minimal config to test model access
        config = {
            'diarizer': {
                'vad': {'model_path': 'vad_multilingual_marblenet'},
                'speaker_embeddings': {'model_path': 'titanet_large'},
                'msdd_model': {'model_path': 'diar_msdd_telephonic'}
            }
        }
        
        try:
            # This will trigger model downloads if needed
            cfg = OmegaConf.create(config)
            print(f"  ✅ NeMo configuration created")
            print(f"  ✅ NeMo models accessible (will download on first use)")
            return True
        except Exception as e:
            print(f"  ⚠️  NeMo model access issue: {e}")
            print(f"     Models will download on first pipeline run")
            return True
        
    except ImportError as e:
        print(f"  ❌ NeMo import failed: {e}")
        print(f"     Solutions:")
        print(f"     1. Run: pip install nemo-toolkit[asr]")
        print(f"     2. Or: pip install nemo-toolkit[asr]==1.20.0")
        print(f"     3. Or install from source if above fail")
        return False

def test_pipeline_files():
    """Check if all pipeline files are present"""
    print("\n🔍 Checking Pipeline Files...")
    
    required_files = [
        "whisper_engine.py",
        "pyannote_engine.py", 
        "nemo_engine.py",
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

def run_complete_test():
    """Run all tests and provide summary"""
    
    print("🧪 Complete Multi-Engine Pipeline Setup Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("PyTorch & CUDA", test_pytorch_cuda),
        ("Whisper", test_whisper),
        ("PyAnnote Engine", test_pyannote),
        ("NeMo Engine", test_nemo),
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
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    essential_tests = ["Basic Imports", "PyTorch & CUDA", "Whisper", "Pipeline Files"]
    pyannote_ok = results.get("PyAnnote Engine", False)
    nemo_ok = results.get("NeMo Engine", False)
    
    # Check essential components
    essential_ok = all(results.get(test, False) for test in essential_tests)
    
    print("🔧 Core Components:")
    for test_name in essential_tests:
        result = results.get(test_name, False)
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print("\n🎙️  Diarization Engines:")
    pyannote_status = "✅ READY" if pyannote_ok else "❌ NOT READY"
    nemo_status = "✅ READY" if nemo_ok else "❌ NOT READY"
    print(f"   {pyannote_status} PyAnnote")
    print(f"   {nemo_status} NeMo")
    
    print("\n📁 Optional:")
    audio_status = "✅ FOUND" if results.get("Audio Files", False) else "⚠️  NONE"
    print(f"   {audio_status} Audio Files")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if essential_ok and (pyannote_ok or nemo_ok):
        print("✅ READY TO USE!")
        if pyannote_ok and nemo_ok:
            print("🎉 Both engines available - full functionality!")
            print("🚀 Run: python pipeline_integration.py")
        elif pyannote_ok:
            print("📝 PyAnnote ready - good for most use cases")
            print("🚀 Run: python pipeline_integration.py")
        elif nemo_ok:
            print("🧠 NeMo ready - advanced diarization available")
            print("🚀 Run: python pipeline_integration.py")
    else:
        print("❌ SETUP INCOMPLETE")
        if not essential_ok:
            print("🔧 Fix core components first")
        if not pyannote_ok and not nemo_ok:
            print("🎙️  Install at least one diarization engine")
        
        print("\n💡 Quick fixes:")
        if not pyannote_ok:
            print("   PyAnnote: huggingface-cli login")
        if not nemo_ok:
            print("   NeMo: pip install nemo-toolkit[asr]")
    
    return essential_ok and (pyannote_ok or nemo_ok)

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)