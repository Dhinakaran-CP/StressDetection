import os
import json
from .artifact_manifest import ArtifactManifest

class VersionRegistry:
    """
    Central registry that keeps track of the currently 'active' versions of all models
    and datasets. The runtime engine will consult this registry to load the correct files
    rather than hardcoding paths.
    """
    def __init__(self, registry_path="backend/expert_models/registry.json"):
        self.registry_path = registry_path
        self._registry = self._load_registry()
        
    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            return {"active_models": {}, "active_datasets": {}}
        with open(self.registry_path, "r") as f:
            return json.load(f)
            
    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=4)
            
    def register_model(self, model_key, manifest: ArtifactManifest):
        """Registers a model manifest as the active version for a given key (e.g., 'face_expert')."""
        self._registry["active_models"][model_key] = manifest.to_dict()
        self._save_registry()
        
    def register_dataset(self, dataset_key, manifest: ArtifactManifest):
        """Registers a dataset manifest as the active version."""
        self._registry["active_datasets"][dataset_key] = manifest.to_dict()
        self._save_registry()

    def get_active_model(self, model_key):
        return self._registry["active_models"].get(model_key)
        
    def get_active_dataset(self, dataset_key):
        return self._registry["active_datasets"].get(dataset_key)
