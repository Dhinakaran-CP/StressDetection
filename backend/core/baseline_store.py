import os
import json

class BaselineStore:
    """
    Manages persistence of user-specific calm baselines in a JSON database schema.
    """
    def __init__(self, store_path="configs/calibrated_baselines.json"):
        self.store_path = store_path
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.baselines = self._load_all_baselines()

    def _load_all_baselines(self):
        if not os.path.exists(self.store_path):
            return {}
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_all_baselines(self):
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.baselines, f, indent=4)

    def get_subject_baseline(self, subject_id):
        """
        Retrieves baseline dict for a subject. Returns None if not calibrated.
        """
        subj_key = str(subject_id).lower().strip()
        return self.baselines.get(subj_key)

    def save_subject_baseline(self, subject_id, face_baseline, voice_baseline, physio_baseline, metadata=None):
        """
        Saves baseline arrays (converted to lists for JSON serialization) for a subject.
        """
        subj_key = str(subject_id).lower().strip()
        
        # Helper to convert numpy arrays/lists cleanly
        def to_list(val):
            if val is None:
                return None
            import numpy as np
            if isinstance(val, np.ndarray):
                return val.tolist()
            return list(val)

        self.baselines[subj_key] = {
            "subject_id": subj_key,
            "face_baseline": to_list(face_baseline),
            "voice_baseline": to_list(voice_baseline),
            "physio_baseline": to_list(physio_baseline),
            "metadata": metadata or {}
        }
        self._save_all_baselines()

    def remove_subject_baseline(self, subject_id):
        """
        Deletes a subject's calibration baseline.
        """
        subj_key = str(subject_id).lower().strip()
        if subj_key in self.baselines:
            del self.baselines[subj_key]
            self._save_all_baselines()
            return True
        return False
