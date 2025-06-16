# test_whisper.py - Simple test script for Whisper setup

import os
import sys
import time
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("=== Checking Dependencies ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        import whisper
        print(f"✓ Whisper: Available")
        
        import librosa
        print(f"✓ Librosa: {librosa.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def check_gpu():
    """Check GPU availability and memory"""
    print("\n=== GPU Information ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU: {gpu_name}")
            print(f"✓ VRAM: {gpu_memory:.1f} GB")
            print(f"✓ CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA not available - will use CPU")
            print("  Make sure you installed PyTorch with CUDA support")
            return False
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False

def test_whisper_basic():
    """Test Whisper model loading"""
    print("\n=== Testing Whisper Model ===")
    
    try:
        import whisper
        
        print("Loading Whisper 'tiny' model for quick test...")
        start_time = time.time()
        
        model = whisper.load_model("tiny")
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"✗ Whisper test failed: {e}")
        return False

def find_audio_files():
    """Find audio files in current directory"""
    print("\n=== Looking for Audio Files ===")
    
    audio_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']
    current_dir = Path('.')
    
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(current_dir.glob(f'*{ext}'))
        audio_files.extend(current_dir.glob(f'*{ext.upper()}'))
    
    if audio_files:
        print("Found audio files:")
        for file in audio_files:
            size_mb = file.stat().st_size / 1e6
            print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
        return audio_files[0]  # Return first file for testing
    else:
        print("✗ No audio files found in current directory")
        print("  Supported formats: WAV, MP3, MP4, M4A, FLAC, OGG")
        return None

def test_full_pipeline(audio_file):
    """Test complete transcription pipeline"""
    print(f"\n=== Testing Full Pipeline ===")
    print(f"Audio file: {audio_file.name}")
    
    try:
        from whisper_engine import WhisperEngine
        
        # Use tiny model for quick test
        print("Initializing Whisper Engine (tiny model)...")
        engine = WhisperEngine(model_size="tiny")
        
        print("Starting transcription...")
        start_time = time.time()
        
        results = engine.transcribe_audio(audio_file)
        
        processing_time = time.time() - start_time
        
        print(f"✓ Transcription completed in {processing_time:.2f} seconds")
        print(f"✓ Detected language: {results['language']}")
        print(f"✓ Text preview: {results['text'][:100]}...")
        
        # Save test output
        engine.save_results(results, "test_output.json", "json")
        print("✓ Output saved to test_output.json")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def recommend_model_size():
    """Recommend model size based on available GPU memory"""
    print("\n=== Model Size Recommendations ===")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory >= 20:
                print(f"✓ {gpu_memory:.1f}GB VRAM - Recommended: 'large' model (best quality)")
            elif gpu_memory >= 10:
                print(f"✓ {gpu_memory:.1f}GB VRAM - Recommended: 'medium' model (good quality)")
            elif gpu_memory >= 4:
                print(f"✓ {gpu_memory:.1f}GB VRAM - Recommended: 'small' model (balanced)")
            else:
                print(f"✓ {gpu_memory:.1f}GB VRAM - Recommended: 'base' model (fast)")
        else:
            print("Using CPU - Recommended: 'base' or 'small' model")
            
    except Exception as e:
        print(f"Could not determine recommendations: {e}")

def main():
    """Main test function"""
    print("🎤 Whisper Local Setup Test")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("pip install -r requirements.txt")
        return False
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Test Whisper
    if not test_whisper_basic():
        return False
    
    # Look for audio files
    audio_file = find_audio_files()
    
    if audio_file:
        # Test full pipeline
        if test_full_pipeline(audio_file):
            print("\n🎉 SUCCESS! Everything is working correctly!")
            recommend_model_size()
            return True
    else:
        print("\n⚠️  Setup is working, but no audio files found for testing")
        print("   Add an audio file to test the complete pipeline")
        recommend_model_size()
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n📋 Next Steps:")
        print("1. Place your audio files in this directory")
        print("2. Run: python whisper_engine.py")
        print("3. Check the 'output/' folder for results")
        print("4. Experiment with different model sizes")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
    
    print("\n" + "=" * 50)