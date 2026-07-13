import time
import threading

class RuntimeMetrics:
    """
    Collects runtime telemetry for the Stress Detection System.
    Tracks latency, missing modality rates, and confidence distribution.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_latency_ms = 0.0
        self.missing_modality_counts = {"face": 0, "voice": 0, "physio": 0}
        self.low_confidence_count = 0
        self.total_confidence = 0.0
        
    def record_prediction(self, latency_ms, missing_modalities, stress_probability):
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += latency_ms
            self.total_confidence += stress_probability
            
            for mod in missing_modalities:
                if mod in self.missing_modality_counts:
                    self.missing_modality_counts[mod] += 1
            
            # We define low confidence as probability near 0.5 (between 0.4 and 0.6)
            if 0.4 <= stress_probability <= 0.6:
                self.low_confidence_count += 1
                
    def get_metrics(self):
        with self._lock:
            avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
            avg_confidence = self.total_confidence / self.total_requests if self.total_requests > 0 else 0
            
            missing_rates = {
                mod: (count / self.total_requests if self.total_requests > 0 else 0)
                for mod, count in self.missing_modality_counts.items()
            }
            
            return {
                "total_requests": self.total_requests,
                "avg_latency_ms": round(avg_latency, 2),
                "missing_modality_rates": {k: round(v, 3) for k, v in missing_rates.items()},
                "low_confidence_ratio": round(self.low_confidence_count / self.total_requests if self.total_requests > 0 else 0, 3),
                "avg_confidence": round(avg_confidence, 3)
            }
