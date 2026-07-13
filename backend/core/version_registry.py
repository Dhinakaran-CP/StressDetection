import os
import json
from .artifact_manifest import ArtifactManifest

class VersionRegistry:
    """
    Central registry that keeps track of the currently active versions of all models,
    datasets, and explainability bundles. The runtime engine consults this registry
    to load correct files rather than hardcoding paths.
    
    In Phase 8, it also maintains a history of registered versions and supports rollbacks.
    """
    def __init__(self, registry_path="models/registry.json"):
        self.registry_path = registry_path
        self._registry = self._load_registry()

    def _load_registry(self):
        if not os.path.exists(self.registry_path):
            return {
                "active_models": {}, 
                "active_datasets": {}, 
                "active_bundles": {},
                "history_models": {},
                "history_datasets": {},
                "history_bundles": {}
            }
        with open(self.registry_path, "r") as f:
            data = json.load(f)
            
        # Back-compat: ensure keys exist in older registry files
        for key in ["active_bundles", "history_models", "history_datasets", "history_bundles"]:
            if key not in data:
                data[key] = {}
        return data

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=4)

    def _append_to_history(self, history_type, key, manifest_dict):
        if key not in self._registry[history_type]:
            self._registry[history_type][key] = []
        
        # Don't append if it's already the latest in history
        history = self._registry[history_type][key]
        if not history or history[-1].get("hash") != manifest_dict.get("hash"):
            history.append(manifest_dict)

    # ── Models ──────────────────────────────────────────────────────────────

    def register_model(self, model_key, manifest: ArtifactManifest):
        """Registers a model manifest as the active version (e.g. 'face_expert')."""
        manifest_dict = manifest.to_dict()
        self._registry["active_models"][model_key] = manifest_dict
        self._append_to_history("history_models", model_key, manifest_dict)
        self._save_registry()

    def get_active_model(self, model_key):
        return self._registry["active_models"].get(model_key)
        
    def rollback_model(self, model_key, version: str) -> bool:
        """Rolls back the active model to a specific version from history."""
        history = self._registry["history_models"].get(model_key, [])
        for past_manifest in reversed(history):
            if past_manifest.get("version") == version:
                self._registry["active_models"][model_key] = past_manifest
                self._save_registry()
                return True
        return False

    # ── Datasets ─────────────────────────────────────────────────────────────

    def register_dataset(self, dataset_key, manifest: ArtifactManifest):
        """Registers a dataset manifest as the active version."""
        manifest_dict = manifest.to_dict()
        self._registry["active_datasets"][dataset_key] = manifest_dict
        self._append_to_history("history_datasets", dataset_key, manifest_dict)
        self._save_registry()

    def get_active_dataset(self, dataset_key):
        return self._registry["active_datasets"].get(dataset_key)

    # ── Explainability Bundles ────────────────────────────────────────────────

    def register_bundle(self, bundle_key, manifest: ArtifactManifest):
        """Registers an explainability bundle manifest as the active version."""
        manifest_dict = manifest.to_dict()
        self._registry["active_bundles"][bundle_key] = manifest_dict
        self._append_to_history("history_bundles", bundle_key, manifest_dict)
        self._save_registry()

    def get_active_bundle(self, bundle_key):
        return self._registry["active_bundles"].get(bundle_key)
