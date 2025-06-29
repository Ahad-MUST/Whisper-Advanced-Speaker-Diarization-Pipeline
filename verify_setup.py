# verify_setup.py - Comprehensive Setup Verification for Speech Diarization Pipeline

import sys
import os
import warnings
import importlib
import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class SetupVerifier:
    """Comprehensive setup verification for the speech diarization pipeline"""
    
    REQUIRED_PACKAGES = {
        # Core packages with minimum versions
        'torch': '2.0.0',
        'torchaudio': '2.0.0',
        'whisper': '1.0.0',
        'speechbrain': '0.5.16',
        'transformers': '4.35.0',
        'librosa': '0.10.1',
        'soundfile': '0.12.1',
        'pandas': '2.1.0',
        'numpy': '1.24.0',
        'scipy': '1.11.0',
        'scikit-learn': '1.3.0',
        'openpyxl': '3.1.2',
        'tqdm': '4.66.0',
        'huggingface_hub': '0.17.0'
    }
    
    OPTIONAL_PACKAGES = {
        'matplotlib': '3.7.0',
        'numba': '0.58.0',
        'datasets': '2.14.0'
    }
    
    def __init__(self):
        self.results = {
            'python_version': False,
            'packages': {},
            'gpu_available': False,
            'whisper_test': False,
            'speechbrain_test': False,
            'audio_processing': False,
            'pipeline_test': False
        }
        self.errors = []
        self.warnings = []
    
    def run_verification(self) -> Dict:
        """Run complete verification process"""
        
        print("=" * 70)
        print("SPEECH DIARIZATION PIPELINE - SETUP VERIFICATION")
        print("=" * 70)
        print()
        
        # Step 1: Check Python version
        self.check_python_version()
        
        # Step 2: Check package installations
        self.check_packages()
        
        # Step 3: Check GPU and CUDA
        self.check_gpu_setup()
        
        # Step 4: Test core engines
        self.test_whisper_engine()
        self.test_speechbrain_engine()
        
        # Step 5: Test audio processing
        self.test_audio_processing()
        
        # Step 6: Test pipeline integration
        self.test_pipeline_integration()
        
        # Step 7: Generate report
        self.generate_report()
        
        return self.results
    
    def check_python_version(self):
        """Check Python version compatibility"""
        print("1. Checking Python version...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major == 3 and version.minor >= 8:
            print(f"   ✓ Python {version_str} - Compatible")
            self.results['python_version'] = True
        else:
            print(f"   ✗ Python {version_str} - Incompatible (requires 3.8+)")
            self.errors.append("Python version too old. Please upgrade to Python 3.8 or newer.")
        
        print()
    
    def check_packages(self):
        """Check all required and optional packages"""
        print("2. Checking package installations...")
        
        # Check required packages
        print("   Required packages:")
        for package, min_version in self.REQUIRED_PACKAGES.items():
            status = self._check_package(package, min_version, required=True)
            self.results['packages'][package] = status
        
        print()
        print("   Optional packages:")
        for package, min_version in self.OPTIONAL_PACKAGES.items():
            status = self._check_package(package, min_version, required=False)
            self.results['packages'][package] = status
        
        print()
    
    def _check_package(self, package_name: str, min_version: str, required: bool = True) -> bool:
        """Check if a specific package is installed with correct version"""
        try:
            # Handle special package name mappings
            import_name = package_name
            if package_name == 'whisper':
                import_name = 'whisper'
            elif package_name == 'huggingface_hub':
                import_name = 'huggingface_hub'
            
            module = importlib.import_module(import_name)
            
            # Get version
            version = getattr(module, '__version__', None)
            if version is None:
                # Try alternative version attributes
                version = getattr(module, 'VERSION', None)
                if version is None:
                    version = "unknown"
            
            # Version comparison (simplified)
            version_ok = True  # Assume OK if we can't compare
            if version != "unknown":
                try:
                    from packaging import version as pkg_version
                    version_ok = pkg_version.parse(version) >= pkg_version.parse(min_version)
                except ImportError:
                    pass  # Skip version check if packaging not available
            
            if version_ok:
                print(f"      ✓ {package_name} {version}")
                return True
            else:
                print(f"      ⚠ {package_name} {version} (requires {min_version}+)")
                if required:
                    self.warnings.append(f"{package_name} version might be too old")
                return False
                
        except ImportError:
            print(f"      ✗ {package_name} - Not installed")
            if required:
                self.errors.append(f"Required package {package_name} is not installed")
            return False
        except Exception as e:
            print(f"      ? {package_name} - Error checking: {e}")
            return False
    
    def check_gpu_setup(self):
        """Check GPU and CUDA availability"""
        print("3. Checking GPU and CUDA setup...")
        
        try:
            import torch
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                
                print(f"   ✓ CUDA available: {cuda_version}")
                print(f"   ✓ GPU devices: {gpu_count}")
                print(f"   ✓ Primary GPU: {gpu_name}")
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   ✓ GPU memory: {gpu_memory:.1f} GB")
                
                if gpu_memory < 4:
                    self.warnings.append("GPU has less than 4GB memory. Large models may not fit.")
                
                self.results['gpu_available'] = True
            else:
                print("   ⚠ CUDA not available - will use CPU")
                print("   ℹ Install CUDA-enabled PyTorch for better performance")
                self.warnings.append("GPU acceleration not available")
                
        except ImportError:
            print("   ✗ PyTorch not available")
            self.errors.append("PyTorch is required but not installed")
        
        print()
    
    def test_whisper_engine(self):
        """Test Whisper engine functionality"""
        print("4. Testing Whisper engine...")
        
        try:
            # Test Whisper import and basic functionality
            import whisper
            import torch
            
            print("   ✓ Whisper import successful")
            
            # Try loading smallest model
            print("   Loading Whisper tiny model...")
            model = whisper.load_model("tiny")
            print("   ✓ Whisper model loading successful")
            
            # Test basic transcription with dummy audio
            print("   Testing transcription...")
            dummy_audio = self._create_dummy_audio()
            
            result = model.transcribe(dummy_audio, verbose=False)
            
            if 'text' in result:
                print("   ✓ Whisper transcription test successful")
                self.results['whisper_test'] = True
            else:
                print("   ⚠ Whisper transcription returned unexpected result")
                
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except ImportError:
            print("   ✗ Whisper import failed")
            self.errors.append("Whisper is not properly installed")
        except Exception as e:
            print(f"   ✗ Whisper test failed: {e}")
            self.errors.append(f"Whisper engine test failed: {e}")
        
        print()
    
    def test_speechbrain_engine(self):
        """Test SpeechBrain engine functionality"""
        print("5. Testing SpeechBrain engine...")
        
        try:
            # Test SpeechBrain import
            import speechbrain
            import torch
            from speechbrain.pretrained import EncoderClassifier
            
            print("   ✓ SpeechBrain import successful")
            
            # Try loading a model (this will download if not cached)
            print("   Loading SpeechBrain model (may download)...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device}
            )
            
            print("   ✓ SpeechBrain model loading successful")
            
            # Test basic embedding extraction
            print("   Testing embedding extraction...")
            dummy_audio = self._create_dummy_audio()
            
            audio_tensor = torch.tensor(dummy_audio).unsqueeze(0)
            if device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            with torch.no_grad():
                embeddings = model.encode_batch(audio_tensor)
                
            if embeddings is not None and embeddings.shape[1] > 0:
                print("   ✓ SpeechBrain embedding extraction successful")
                self.results['speechbrain_test'] = True
            else:
                print("   ⚠ SpeechBrain embedding extraction returned unexpected result")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except ImportError:
            print("   ✗ SpeechBrain import failed")
            self.errors.append("SpeechBrain is not properly installed")
        except Exception as e:
            print(f"   ✗ SpeechBrain test failed: {e}")
            self.errors.append(f"SpeechBrain engine test failed: {e}")
        
        print()
    
    def test_audio_processing(self):
        """Test audio processing capabilities"""
        print("6. Testing audio processing...")
        
        try:
            # Test librosa
            import librosa
            import soundfile as sf
            import numpy as np
            
            print("   ✓ Audio libraries import successful")
            
            # Test audio generation and processing
            dummy_audio = self._create_dummy_audio()
            
            # Test resampling
            resampled = librosa.resample(dummy_audio, orig_sr=16000, target_sr=8000)
            print("   ✓ Audio resampling test successful")
            
            # Test file I/O
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, dummy_audio, 16000)
                loaded_audio, sr = librosa.load(tmp_file.name, sr=16000)
                os.unlink(tmp_file.name)
            
            print("   ✓ Audio file I/O test successful")
            
            # Test pandas for Excel output
            import pandas as pd
            df = pd.DataFrame({'test': [1, 2, 3]})
            
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                df.to_excel(tmp_file.name, index=False)
                os.unlink(tmp_file.name)
            
            print("   ✓ Excel output test successful")
            
            self.results['audio_processing'] = True
            
        except ImportError as e:
            print(f"   ✗ Audio processing import failed: {e}")
            self.errors.append(f"Audio processing libraries not properly installed: {e}")
        except Exception as e:
            print(f"   ✗ Audio processing test failed: {e}")
            self.errors.append(f"Audio processing test failed: {e}")
        
        print()
    
    def test_pipeline_integration(self):
        """Test if the main pipeline files can be imported"""
        print("7. Testing pipeline integration...")
        
        # Check if main pipeline files exist and can be imported
        pipeline_files = [
            'whisper_engine.py',
            'speechbrain_engine.py', 
            'basic_audio_preprocessor.py',
            'run.py'
        ]
        
        missing_files = []
        import_errors = []
        
        for file in pipeline_files:
            if not Path(file).exists():
                missing_files.append(file)
                continue
                
            # Try to import the module
            try:
                module_name = file.replace('.py', '')
                spec = importlib.util.spec_from_file_location(module_name, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"   ✓ {file} import successful")
            except Exception as e:
                print(f"   ✗ {file} import failed: {e}")
                import_errors.append(f"{file}: {e}")
        
        if missing_files:
            print(f"   ✗ Missing files: {', '.join(missing_files)}")
            self.errors.append(f"Missing pipeline files: {', '.join(missing_files)}")
        
        if import_errors:
            self.errors.extend(import_errors)
        
        if not missing_files and not import_errors:
            print("   ✓ All pipeline files available and importable")
            self.results['pipeline_test'] = True
        
        print()
    
    def _create_dummy_audio(self) -> 'np.ndarray':
        """Create dummy audio for testing"""
        import numpy as np
        
        # Create 1 second of dummy audio (sine wave)
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        return audio.astype(np.float32)
    
    def generate_report(self):
        """Generate final verification report"""
        print("=" * 70)
        print("VERIFICATION REPORT")
        print("=" * 70)
        
        # Count successful tests
        total_tests = 7
        passed_tests = sum([
            self.results['python_version'],
            len([p for p in self.results['packages'].values() if p]) >= len(self.REQUIRED_PACKAGES) * 0.8,
            self.results['whisper_test'],
            self.results['speechbrain_test'],
            self.results['audio_processing'],
            self.results['pipeline_test']
        ])
        
        print(f"\nOVERALL STATUS: {passed_tests}/{total_tests} tests passed")
        
        # System readiness assessment
        required_passed = all([
            self.results['python_version'],
            self.results['whisper_test'],
            self.results['speechbrain_test'],
            self.results['audio_processing'],
            self.results['pipeline_test']
        ])
        
        if required_passed:
            print("\n✓ SYSTEM READY - All core components are working!")
            if self.results['gpu_available']:
                print("✓ GPU acceleration available for optimal performance")
            else:
                print("⚠ CPU-only mode (consider installing CUDA for better performance)")
        else:
            print("\n✗ SYSTEM NOT READY - Critical issues found")
        
        # Show errors
        if self.errors:
            print(f"\nCRITICAL ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")
        
        # Show warnings
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        
        if not self.results['python_version']:
            print("   • Upgrade to Python 3.8 or newer")
        
        missing_packages = [pkg for pkg, status in self.results['packages'].items() 
                          if not status and pkg in self.REQUIRED_PACKAGES]
        if missing_packages:
            print("   • Install missing packages:")
            print(f"     pip install {' '.join(missing_packages)}")
        
        if not self.results['gpu_available']:
            print("   • Install CUDA-enabled PyTorch for GPU acceleration:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        if self.errors:
            print("   • Fix critical errors before running the pipeline")
            print("   • Reinstall packages if import errors persist")
            print("   • Check internet connection for model downloads")
        
        print("\nTo run the pipeline after fixing issues:")
        print("   python run.py")
        
        print("\n" + "=" * 70)


def main():
    """Main verification function"""
    try:
        # Add current directory to Python path for imports
        sys.path.insert(0, os.getcwd())
        
        # Run verification
        verifier = SetupVerifier()
        results = verifier.run_verification()
        
        # Exit with appropriate code
        if len(verifier.errors) == 0:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Errors found
            
    except KeyboardInterrupt:
        print("\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nVerification script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()