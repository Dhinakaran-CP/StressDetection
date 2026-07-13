import unittest
import numpy as np
import time

from backend.monitoring.runtime_metrics import RuntimeMetrics
from backend.monitoring.drift_monitor import DriftMonitor
from backend.monitoring.golden_replay import GoldenReplay
from backend.core.version_registry import VersionRegistry
from backend.core.artifact_manifest import ArtifactManifest
from backend.runtime.runtime_engine import RuntimeEngine

class TestMonitoringAndRollback(unittest.TestCase):
    
    def test_runtime_metrics(self):
        metrics = RuntimeMetrics()
        metrics.record_prediction(latency_ms=10.5, missing_modalities=["voice"], stress_probability=0.45)
        metrics.record_prediction(latency_ms=15.0, missing_modalities=[], stress_probability=0.8)
        
        report = metrics.get_metrics()
        self.assertEqual(report["total_requests"], 2)
        self.assertAlmostEqual(report["avg_latency_ms"], 12.75)
        self.assertEqual(report["missing_modality_rates"]["voice"], 0.5)
        self.assertEqual(report["low_confidence_ratio"], 0.5)
        self.assertAlmostEqual(report["avg_confidence"], 0.625)

    def test_drift_monitor(self):
        drift = DriftMonitor(window_size=2)
        drift.record_features(face=np.array([1.0, 2.0]))
        drift.record_features(face=np.array([3.0, 4.0]))
        
        report = drift.get_drift_report()
        self.assertEqual(report["face"]["samples"], 2)
        self.assertEqual(report["face"]["rolling_mean"], [2.0, 3.0])
        
        # Test window eviction
        drift.record_features(face=np.array([5.0, 6.0]))
        report = drift.get_drift_report()
        self.assertEqual(report["face"]["samples"], 2)
        self.assertEqual(report["face"]["rolling_mean"], [4.0, 5.0])

    def test_golden_replay(self):
        engine = RuntimeEngine.from_registry()
        if not engine.is_ready:
            self.skipTest("Engine not ready, skipping.")
            
        replay = GoldenReplay(engine)
        
        # Use dummy rows
        rows = [
            {"face": np.random.rand(18).tolist(), "voice": None, "physio": np.random.rand(5).tolist()}
        ]
        
        res = replay.run_replay(rows)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["count"], 1)
        self.assertIn("stress_probability", res["results"][0])

    def test_version_registry_rollback(self):
        registry = VersionRegistry(registry_path="backend/tests/dummy_registry.json")
        registry._registry = {
            "active_models": {}, "active_datasets": {}, "active_bundles": {},
            "history_models": {}, "history_datasets": {}, "history_bundles": {}
        }
        
        # Register v1
        m1 = ArtifactManifest(artifact_id="test_model", artifact_type="model", version="1.0.0")
        m1.hash = "hash1"
        registry.register_model("face", m1)
        
        # Register v2
        m2 = ArtifactManifest(artifact_id="test_model", artifact_type="model", version="2.0.0")
        m2.hash = "hash2"
        registry.register_model("face", m2)
        
        self.assertEqual(registry.get_active_model("face")["version"], "2.0.0")
        self.assertEqual(len(registry._registry["history_models"]["face"]), 2)
        
        # Rollback to v1
        success = registry.rollback_model("face", "1.0.0")
        self.assertTrue(success)
        self.assertEqual(registry.get_active_model("face")["version"], "1.0.0")
        
        # Try rollback to invalid
        success = registry.rollback_model("face", "3.0.0")
        self.assertFalse(success)
        
        import os
        if os.path.exists("backend/tests/dummy_registry.json"):
            os.remove("backend/tests/dummy_registry.json")

if __name__ == "__main__":
    unittest.main()
