# cudnn_diagnostic.py - Comprehensive CUDNN installation diagnostic

import os
import sys
import subprocess
from pathlib import Path
import torch

def find_dll_files(directory, pattern):
    """Find DLL files matching pattern in directory"""
    try:
        path = Path(directory)
        if path.exists():
            return list(path.glob(pattern))
    except:
        pass
    return []

def check_python_info():
    """Check Python and PyTorch installation"""
    print("🐍 PYTHON & PYTORCH INFO")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"CUDNN Version: {torch.backends.cudnn.version()}")
    print(f"CUDNN Enabled: {torch.backends.cudnn.enabled}")
    print()

def check_cudnn_installations():
    """Find all CUDNN installations"""
    print("🔍 CUDNN INSTALLATION LOCATIONS")
    print("=" * 50)
    
    locations_to_check = [
        # CUDA Toolkit locations
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin", 
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7/bin",
        # System directories
        "C:/Windows/System32",
        "C:/Windows/SysWOW64",
        # Python directories
        str(Path(sys.executable).parent),
        # Pip installed CUDNN
        str(Path(sys.executable).parent / "Lib/site-packages/nvidia/cudnn/bin"),
    ]
    
    cudnn_found = []
    
    for location in locations_to_check:
        print(f"📂 Checking: {location}")
        
        # Check for various CUDNN DLL patterns
        patterns = ["cudnn*.dll", "cudnn64*.dll", "*cudnn*.dll"]
        found_in_location = []
        
        for pattern in patterns:
            dlls = find_dll_files(location, pattern)
            for dll in dlls:
                if dll.name not in [f.name for f in found_in_location]:
                    found_in_location.append(dll)
        
        if found_in_location:
            for dll in found_in_location:
                version = "Unknown"
                if "cudnn64_8" in dll.name or "cudnn_8" in dll.name:
                    version = "8.x"
                elif "cudnn64_9" in dll.name or "cudnn_9" in dll.name:
                    version = "9.x"
                elif "disabled" in dll.name:
                    version = "Disabled"
                
                print(f"   ✅ Found: {dll.name} (Version: {version})")
                cudnn_found.append({
                    'path': str(dll),
                    'name': dll.name,
                    'version': version,
                    'location': location
                })
        else:
            print(f"   ❌ No CUDNN files found")
    
    print()
    return cudnn_found

def check_environment():
    """Check environment variables"""
    print("🌍 ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    important_vars = ['PATH', 'CUDA_PATH', 'CUDNN_PATH', 'LD_LIBRARY_PATH']
    
    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    print()

def test_whisperx():
    """Test WhisperX import"""
    print("🧪 WHISPERX TEST")
    print("=" * 50)
    
    try:
        import whisperx
        print("✅ WhisperX import: SUCCESS")
        
        try:
            # Try to load a small model
            print("🔄 Testing WhisperX model loading...")
            model = whisperx.load_model("tiny", device="cuda", compute_type="float16")
            print("✅ WhisperX model loading: SUCCESS")
            return True
        except Exception as e:
            print(f"❌ WhisperX model loading: FAILED")
            print(f"   Error: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ WhisperX import: FAILED")
        print(f"   Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ WhisperX test: FAILED")
        print(f"   Error: {str(e)}")
        return False

def generate_fix_commands(cudnn_found):
    """Generate fix commands based on findings"""
    print("🔧 RECOMMENDED FIXES")
    print("=" * 50)
    
    cudnn_9_files = [f for f in cudnn_found if f['version'] == '9.x']
    cudnn_8_files = [f for f in cudnn_found if f['version'] == '8.x']
    
    if not cudnn_8_files:
        print("❌ CUDNN 8.x not found!")
        print("📥 Install CUDNN 8.x:")
        print("   pip install nvidia-cudnn-cu12==8.9.7.29")
        print()
    
    if cudnn_9_files:
        print("❌ CUDNN 9.x files found (conflicting):")
        for f in cudnn_9_files:
            print(f"   {f['path']}")
        print()
        print("🔧 To disable CUDNN 9.x files:")
        for f in cudnn_9_files:
            disabled_name = f['name'].replace('.dll', '.dll.disabled')
            print(f"   ren \"{f['path']}\" \"{disabled_name}\"")
        print()
    
    if cudnn_8_files:
        print("✅ CUDNN 8.x files found:")
        for f in cudnn_8_files:
            print(f"   {f['path']}")
        
        # Check if 8.x files are in Python directory
        python_dir = str(Path(sys.executable).parent)
        cudnn_8_in_python = any(python_dir in f['path'] for f in cudnn_8_files)
        
        if not cudnn_8_in_python:
            print()
            print("🔧 Copy CUDNN 8.x to Python directory:")
            pip_cudnn_path = str(Path(sys.executable).parent / "Lib/site-packages/nvidia/cudnn/bin")
            print(f"   copy \"{pip_cudnn_path}\\*.dll\" \"{python_dir}\\\"")
    
    print()
    print("🧪 Test after fixes:")
    print("   python -c \"import torch; print(f'CUDNN: {torch.backends.cudnn.version()}')\"")
    print("   Expected result: 8907 (for CUDNN 8.x)")

def main():
    """Main diagnostic function"""
    print("🔍 CUDNN DIAGNOSTIC REPORT")
    print("=" * 70)
    print("This script will help diagnose CUDNN installation issues")
    print("=" * 70)
    print()
    
    # Check Python and PyTorch
    check_python_info()
    
    # Check environment
    check_environment()
    
    # Find CUDNN installations
    cudnn_found = check_cudnn_installations()
    
    # Test WhisperX
    whisperx_works = test_whisperx()
    
    # Generate fix recommendations
    generate_fix_commands(cudnn_found)
    
    print("🎯 SUMMARY")
    print("=" * 50)
    current_cudnn = torch.backends.cudnn.version()
    
    if current_cudnn >= 90000:
        print(f"❌ Currently using CUDNN 9.x ({current_cudnn})")
        print("🔧 Need to disable CUDNN 9.x and enable CUDNN 8.x")
    elif current_cudnn >= 80000:
        print(f"✅ Currently using CUDNN 8.x ({current_cudnn})")
        if whisperx_works:
            print("✅ WhisperX should work!")
        else:
            print("⚠️  WhisperX still has issues - check error messages above")
    else:
        print(f"⚠️  Unusual CUDNN version: {current_cudnn}")
    
    print()
    print("📋 NEXT STEPS:")
    if current_cudnn >= 90000:
        print("1. Run the cudnn_fix.bat script")
        print("2. Or manually execute the fix commands above")
        print("3. Test: python -c \"import torch; print(f'CUDNN: {torch.backends.cudnn.version()}')\"")
        print("4. Run your pipeline: python multi_engine_pipeline.py")
    else:
        print("1. Test your pipeline: python multi_engine_pipeline.py")
        print("2. Select WhisperX (option 2) - should work now!")

if __name__ == "__main__":
    main()