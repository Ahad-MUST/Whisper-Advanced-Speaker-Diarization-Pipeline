# setup_whisperx_env.py - Automated WhisperX Environment Setup
# Run this script to automatically set up your WhisperX virtual environment

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a shell command with error handling"""
    print(f"🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(f"   ✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def detect_cuda_version():
    """Detect CUDA version if available"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
        if "CUDA Version: 12" in result.stdout:
            return "cu121"
        elif "CUDA Version: 11" in result.stdout:
            return "cu118"
        else:
            return "cu118"  # Default to 11.8
    except:
        return "cpu"

def get_pytorch_install_command():
    """Get appropriate PyTorch installation command"""
    cuda_version = detect_cuda_version()
    
    if cuda_version == "cpu":
        return "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    elif cuda_version == "cu121":
        return "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        return "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"

def create_project_structure():
    """Create recommended project structure"""
    directories = [
        "audio_input",
        "output",
        "models",
        "logs",
        "scripts"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   📁 Created directory: {dir_name}")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
whisperx_env/

# Audio files
*.mp3
*.wav
*.mp4
*.m4a
*.flac
*.ogg

# Output files
output/
logs/
*.log

# Models (large files)
models/
*.pt
*.bin
*.safetensors

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    print("   📄 Created .gitignore")

def main():
    print("🚀 WhisperX Virtual Environment Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required!")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Detect OS
    os_name = platform.system()
    print(f"✅ Operating System: {os_name}")
    
    # Detect CUDA
    cuda_version = detect_cuda_version()
    print(f"🎯 CUDA Detection: {cuda_version}")
    
    # Create virtual environment
    env_name = "whisperx_env"
    
    if not Path(env_name).exists():
        print(f"\n📦 Creating virtual environment: {env_name}")
        if not run_command(f"python -m venv {env_name}", "Creating virtual environment"):
            sys.exit(1)
    else:
        print(f"✅ Virtual environment already exists: {env_name}")
    
    # Activation commands based on OS
    if os_name == "Windows":
        activate_cmd = f"{env_name}\\Scripts\\activate"
        pip_cmd = f"{env_name}\\Scripts\\pip"
        python_cmd = f"{env_name}\\Scripts\\python"
    else:
        activate_cmd = f"source {env_name}/bin/activate"
        pip_cmd = f"{env_name}/bin/pip"
        python_cmd = f"{env_name}/bin/python"
    
    print(f"\n🔧 Installing packages...")
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip setuptools wheel", "Upgrading pip")
    
    # Install PyTorch
    pytorch_cmd = get_pytorch_install_command().replace("pip", pip_cmd)
    if not run_command(pytorch_cmd, "Installing PyTorch"):
        print("⚠️  PyTorch installation failed, continuing...")
    
    # Install main packages
    packages = [
        "whisperx",
        "pyannote.audio",
        "speechbrain",
        "scikit-learn",
        "pandas",
        "openpyxl",
        "numpy",
        "librosa",
        "soundfile",
        "huggingface_hub",
        "transformers",
        "tqdm"
    ]
    
    for package in packages:
        run_command(f"{pip_cmd} install {package}", f"Installing {package}", check=False)
    
    # Test installation
    print(f"\n🧪 Testing installation...")
    
    test_script = '''
import sys
try:
    import whisperx
    print("✅ WhisperX")
except ImportError as e:
    print(f"❌ WhisperX: {e}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import pyannote.audio
    print("✅ PyAnnote")
except ImportError as e:
    print(f"❌ PyAnnote: {e}")

try:
    import speechbrain
    print("✅ SpeechBrain")
except ImportError as e:
    print(f"❌ SpeechBrain: {e}")

try:
    import pandas as pd
    print("✅ Pandas")
except ImportError as e:
    print(f"❌ Pandas: {e}")
'''
    
    run_command(f'{python_cmd} -c "{test_script}"', "Testing packages", check=False)
    
    # Create project structure
    print(f"\n📁 Creating project structure...")
    create_project_structure()
    
    # Create activation scripts
    print(f"\n📜 Creating convenience scripts...")
    
    if os_name == "Windows":
        # Windows batch file
        with open("activate_env.bat", "w") as f:
            f.write(f"@echo off\n")
            f.write(f"call {env_name}\\Scripts\\activate\n")
            f.write(f"echo ✅ WhisperX environment activated!\n")
            f.write(f"echo Run: python whisperx_diarization_standalone.py\n")
        print("   📄 Created activate_env.bat")
        
        # PowerShell script
        with open("activate_env.ps1", "w") as f:
            f.write(f"& .\\{env_name}\\Scripts\\Activate.ps1\n")
            f.write(f'Write-Host "✅ WhisperX environment activated!" -ForegroundColor Green\n')
            f.write(f'Write-Host "Run: python whisperx_diarization_standalone.py" -ForegroundColor Yellow\n')
        print("   📄 Created activate_env.ps1")
    else:
        # Unix shell script
        with open("activate_env.sh", "w") as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"source {env_name}/bin/activate\n")
            f.write(f'echo "✅ WhisperX environment activated!"\n')
            f.write(f'echo "Run: python whisperx_diarization_standalone.py"\n')
        
        # Make executable
        os.chmod("activate_env.sh", 0o755)
        print("   📄 Created activate_env.sh")
    
    # Final instructions
    print(f"\n🎉 Setup Complete!")
    print(f"=" * 50)
    print(f"📁 Project directory: {Path.cwd()}")
    print(f"🐍 Virtual environment: {env_name}")
    print()
    print(f"🚀 To activate environment:")
    
    if os_name == "Windows":
        print(f"   Option 1: {env_name}\\Scripts\\activate")
        print(f"   Option 2: .\\activate_env.bat")
        print(f"   Option 3: .\\activate_env.ps1")
    else:
        print(f"   Option 1: source {env_name}/bin/activate")
        print(f"   Option 2: source activate_env.sh")
    
    print()
    print(f"📋 Next steps:")
    print(f"   1. Activate the environment")
    print(f"   2. Copy your WhisperX script to this directory")
    print(f"   3. For PyAnnote: run 'huggingface-cli login'")
    print(f"   4. Put audio files in 'audio_input/' directory")
    print(f"   5. Run: python whisperx_diarization_standalone.py")
    print()
    print(f"📁 Directory structure created:")
    print(f"   audio_input/  - Put your audio files here")
    print(f"   output/       - Processed results go here")
    print(f"   models/       - Downloaded models cache")
    print(f"   logs/         - Log files")
    print(f"   scripts/      - Additional scripts")

if __name__ == "__main__":
    main()