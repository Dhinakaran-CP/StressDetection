import os
import json
import pandas as pd
from backend.core.dataset_certifier import DatasetCertifier
from backend.core.artifact_manifest import ArtifactManifest
from backend.core.version_registry import VersionRegistry

def release_datasets():
    """
    Reads the raw datasets, certifies them, computes hashes, and releases them 
    into the 'dataset_certified/' directory.
    """
    RAW_DIR = "dataset_extracted"
    CERTIFIED_DIR = "dataset_certified"
    os.makedirs(CERTIFIED_DIR, exist_ok=True)
    
    face_raw_path = os.path.join(RAW_DIR, "face_indicators_stressid.csv")
    voice_raw_path = os.path.join(RAW_DIR, "voice_indicators_stressid.csv")
    
    certifier = DatasetCertifier()
    registry = VersionRegistry()
    
    # 1. Load Data
    print("Loading raw CSV files...")
    face_df = pd.read_csv(face_raw_path)
    voice_df = pd.read_csv(voice_raw_path)
    
    # 2. Certify Data
    face_report = certifier.validate(face_df, "Face Modality")
    voice_report = certifier.validate(voice_df, "Voice Modality")
    
    # 3. Save Certified Copies
    face_cert_path = os.path.join(CERTIFIED_DIR, "face_certified.csv")
    voice_cert_path = os.path.join(CERTIFIED_DIR, "voice_certified.csv")
    
    face_df.to_csv(face_cert_path, index=False)
    voice_df.to_csv(voice_cert_path, index=False)
    
    # 4. Generate Manifests and Register
    face_manifest = ArtifactManifest("face_dataset_v1", "dataset", "1.0.0", metadata=face_report)
    face_manifest.compute_hash(face_cert_path)
    face_manifest.save(face_cert_path)
    
    voice_manifest = ArtifactManifest("voice_dataset_v1", "dataset", "1.0.0", metadata=voice_report)
    voice_manifest.compute_hash(voice_cert_path)
    voice_manifest.save(voice_cert_path)
    
    registry.register_dataset("face_dataset", face_manifest)
    registry.register_dataset("voice_dataset", voice_manifest)
    
    print("\n==============================================")
    print("DATASET RELEASE COMPLETE")
    print(f"Face Dataset Hash: {face_manifest.hash}")
    print(f"Voice Dataset Hash: {voice_manifest.hash}")
    print("==============================================")

if __name__ == "__main__":
    release_datasets()
