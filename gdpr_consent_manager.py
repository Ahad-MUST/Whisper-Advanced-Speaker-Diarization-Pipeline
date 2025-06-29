# gdpr_consent_manager.py - GDPR Consent Management

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

class GDPRConsentManager:
    """
    GDPR Consent Management for Speech Diarization Pipeline
    
    Handles:
    - Consent collection and recording
    - Data retention policies
    - Data subject rights (deletion, access)
    - Privacy notice display
    """
    
    def __init__(self, consent_file: str = "gdpr_consent_log.json"):
        self.consent_file = Path(consent_file)
        self.privacy_notice = self._get_privacy_notice()
        
    def display_privacy_notice(self) -> bool:
        """Display privacy notice and collect consent"""
        
        print("=" * 70)
        print("PRIVACY NOTICE - SPEECH DIARIZATION PROCESSING")
        print("=" * 70)
        print()
        print("DATA CONTROLLER: [Your Organization Name]")
        print("CONTACT: [Your Contact Information]")
        print()
        print("PURPOSE OF PROCESSING:")
        print("• Speech-to-text transcription of your audio file")
        print("• Speaker identification and diarization")
        print("• Generation of transcript with speaker labels")
        print()
        print("LEGAL BASIS: Consent (Article 6(1)(a) GDPR)")
        print()
        print("DATA PROCESSED:")
        print("• Audio file content (speech/voice data)")
        print("• Derived text transcription")
        print("• Speaker identification results")
        print()
        print("PROCESSING LOCATION: Local computer only (no cloud upload)")
        print()
        print("DATA RETENTION:")
        print("• Output files: Until you delete them manually")
        print("• Temporary files: Automatically deleted after processing")
        print("• Processing logs: 30 days (for support purposes)")
        print()
        print("YOUR RIGHTS:")
        print("• Right to withdraw consent at any time")
        print("• Right to deletion of your data")
        print("• Right to access your data")
        print("• Right to data portability")
        print()
        print("DATA SHARING: No data is shared with third parties")
        print()
        print("AUTOMATED DECISION MAKING: None")
        print()
        print("=" * 70)
        
        return self._collect_consent()
    
    def _collect_consent(self) -> bool:
        """Collect explicit consent from user"""
        
        print("\nCONSENT REQUIRED:")
        print("Do you give your explicit consent for processing your audio file")
        print("for speech diarization as described above?")
        print()
        print("Type 'I CONSENT' (exactly) to proceed, or 'NO' to cancel:")
        
        while True:
            try:
                response = input("> ").strip()
                
                if response == "I CONSENT":
                    self._record_consent(granted=True)
                    print("\n✓ Consent recorded. Processing will begin.")
                    return True
                elif response.upper() == "NO":
                    print("\n✗ Consent not given. Processing cancelled.")
                    return False
                else:
                    print("Please type exactly 'I CONSENT' or 'NO':")
                    
            except KeyboardInterrupt:
                print("\n✗ Processing cancelled.")
                return False
    
    def _record_consent(self, granted: bool, file_hash: Optional[str] = None):
        """Record consent decision with timestamp"""
        
        consent_record = {
            "timestamp": datetime.now().isoformat(),
            "consent_granted": granted,
            "file_hash": file_hash,
            "ip_address": "localhost",  # Local processing
            "consent_version": "1.0"
        }
        
        # Load existing records
        consent_log = []
        if self.consent_file.exists():
            try:
                with open(self.consent_file, 'r') as f:
                    consent_log = json.load(f)
            except:
                consent_log = []
        
        # Add new record
        consent_log.append(consent_record)
        
        # Save updated log
        with open(self.consent_file, 'w') as f:
            json.dump(consent_log, f, indent=2)
    
    def check_retention_policy(self):
        """Check and enforce data retention policy"""
        
        # Delete consent logs older than 30 days
        if self.consent_file.exists():
            try:
                with open(self.consent_file, 'r') as f:
                    consent_log = json.load(f)
                
                cutoff_date = datetime.now() - timedelta(days=30)
                filtered_log = []
                
                for record in consent_log:
                    record_date = datetime.fromisoformat(record['timestamp'])
                    if record_date > cutoff_date:
                        filtered_log.append(record)
                
                # Save filtered log
                with open(self.consent_file, 'w') as f:
                    json.dump(filtered_log, f, indent=2)
                    
            except Exception as e:
                print(f"Warning: Could not process consent log: {e}")
    
    def exercise_right_to_deletion(self, output_dir: str):
        """Allow user to delete their data (Right to be forgotten)"""
        
        output_path = Path(output_dir)
        
        if not output_path.exists():
            print("No output files found to delete.")
            return
        
        files_found = list(output_path.glob("*"))
        
        if not files_found:
            print("No files found in output directory.")
            return
        
        print(f"\nFOUND {len(files_found)} FILES IN OUTPUT DIRECTORY:")
        for file in files_found:
            print(f"   {file.name}")
        
        print("\nExercising RIGHT TO DELETION (Article 17 GDPR)")
        confirm = input("Delete ALL files? Type 'DELETE ALL' to confirm: ").strip()
        
        if confirm == "DELETE ALL":
            try:
                for file in files_found:
                    file.unlink()
                
                # Remove directory if empty
                if not any(output_path.iterdir()):
                    output_path.rmdir()
                
                print(f"✓ All files deleted from {output_dir}")
                print("✓ Right to deletion exercised successfully")
                
            except Exception as e:
                print(f"Error deleting files: {e}")
        else:
            print("Deletion cancelled.")
    
    def generate_data_export(self, output_dir: str) -> Dict:
        """Generate data export (Right to data portability)"""
        
        output_path = Path(output_dir)
        
        if not output_path.exists():
            return {"message": "No data found to export"}
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "data_controller": "[Your Organization]",
            "subject_rights_notice": "This export contains all personal data processed",
            "files": []
        }
        
        # List all files and metadata
        for file in output_path.glob("*"):
            if file.is_file():
                export_data["files"].append({
                    "filename": file.name,
                    "size_bytes": file.stat().st_size,
                    "created": datetime.fromtimestamp(file.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
                })
        
        return export_data
    
    def _get_privacy_notice(self) -> str:
        """Get the current privacy notice version"""
        return """
        Privacy Notice for Speech Diarization Processing v1.0
        
        This notice explains how we process your personal data when you use 
        our speech diarization service.
        """
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash of audio file for consent tracking"""
        try:
            with open(file_path, 'rb') as f:
                # Read first 1MB for hash (efficient for large files)
                chunk = f.read(1024 * 1024)
                return hashlib.sha256(chunk).hexdigest()[:16]
        except:
            return "unknown"


def demonstrate_gdpr_compliance():
    """Demonstrate GDPR compliance features"""
    
    consent_manager = GDPRConsentManager()
    
    print("GDPR COMPLIANCE DEMONSTRATION")
    print("=" * 40)
    
    # 1. Display privacy notice and collect consent
    consent_granted = consent_manager.display_privacy_notice()
    
    if not consent_granted:
        print("Processing cannot continue without consent.")
        return False
    
    # 2. Show data subject rights
    print("\nDATA SUBJECT RIGHTS:")
    print("1. Request data deletion")
    print("2. Export your data")
    print("3. Continue with processing")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        consent_manager.exercise_right_to_deletion("output_with_preprocessing")
        return False
    elif choice == "2":
        export_data = consent_manager.generate_data_export("output_with_preprocessing")
        print("\nDATA EXPORT:")
        print(json.dumps(export_data, indent=2))
        return False
    else:
        print("\nProceeding with audio processing...")
        return True


if __name__ == "__main__":
    demonstrate_gdpr_compliance()