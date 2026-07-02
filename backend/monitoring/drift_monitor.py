import numpy as np
import threading

class DriftMonitor:
    """
    Monitors incoming features for distribution drift against baseline metrics.
    In Phase 8, this maintains rolling means of feature vectors.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self._lock = threading.Lock()
        
        self.buffers = {
            "face": [],
            "voice": [],
            "physio": []
        }
        
    def record_features(self, face=None, voice=None, physio=None):
        """Records features for drift analysis."""
        with self._lock:
            if face is not None:
                self.buffers["face"].append(face)
                if len(self.buffers["face"]) > self.window_size:
                    self.buffers["face"].pop(0)
                    
            if voice is not None:
                self.buffers["voice"].append(voice)
                if len(self.buffers["voice"]) > self.window_size:
                    self.buffers["voice"].pop(0)
                    
            if physio is not None:
                self.buffers["physio"].append(physio)
                if len(self.buffers["physio"]) > self.window_size:
                    self.buffers["physio"].pop(0)
                    
    def get_drift_report(self):
        """Returns the current rolling means for each modality."""
        report = {}
        with self._lock:
            for mod, buf in self.buffers.items():
                if len(buf) > 0:
                    # Compute mean across the buffer
                    rolling_mean = np.mean(buf, axis=0)
                    report[mod] = {
                        "samples": len(buf),
                        "rolling_mean": rolling_mean.tolist()
                    }
                else:
                    report[mod] = {"samples": 0, "rolling_mean": None}
        return report
