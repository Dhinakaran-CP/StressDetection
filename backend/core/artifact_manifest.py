import os
import json
import hashlib
import time

class ArtifactManifest:
    """
    Represents a versioned ML artifact (model, scaler, dataset, explainability bundle).
    This enforces the Phase 1 contracts by making sure everything has a traceable hash,
    creation time, and version string.
    """
    def __init__(self, artifact_id, artifact_type, version, metadata=None):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type  # 'model', 'dataset', 'scaler', 'bundle'
        self.version = version
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.hash = None
        
    def compute_hash(self, file_path):
        """Computes SHA256 of the physical file to lock the manifest to the exact binary."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot hash missing artifact: {file_path}")
            
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        self.hash = sha256.hexdigest()
        return self.hash

    def to_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "hash": self.hash,
            "metadata": self.metadata
        }
        
    def save(self, output_path):
        """Saves the manifest alongside the artifact as a .json file."""
        manifest_path = f"{output_path}.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        return manifest_path

    @classmethod
    def load(cls, manifest_path):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            
        with open(manifest_path, "r") as f:
            data = json.load(f)
            
        obj = cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            version=data["version"],
            metadata=data.get("metadata", {})
        )
        obj.timestamp = data["timestamp"]
        obj.hash = data["hash"]
        return obj
