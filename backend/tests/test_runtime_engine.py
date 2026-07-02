import unittest
import numpy as np
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.runtime.runtime_engine import RuntimeEngine
from backend.runtime.session_state  import SessionState

class TestSessionState(unittest.TestCase):
    def test_buffer_audio_short(self):
        """Should return None if audio length < MIN threshold."""
        session = SessionState("test_1", min_audio_seconds=0.5, sample_rate=16000)
        # 0.25 seconds
        chunk = np.zeros(int(16000 * 0.25), dtype=np.float32)
        result = session.buffer_audio(chunk)
        self.assertIsNone(result)
        self.assertEqual(session.audio_duration_seconds, 0.25)

    def test_buffer_audio_ready(self):
        """Should return full buffer if >= MIN threshold."""
        session = SessionState("test_2", min_audio_seconds=0.5, sample_rate=16000)
        chunk = np.zeros(int(16000 * 0.6), dtype=np.float32)
        result = session.buffer_audio(chunk)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), int(16000 * 0.6))
        self.assertEqual(session.audio_duration_seconds, 0.6)

    def test_buffer_audio_max_cap(self):
        """Should truncate older audio keeping only MAX seconds."""
        session = SessionState("test_3", max_audio_seconds=1.0, min_audio_seconds=0.5, sample_rate=16000)
        chunk1 = np.ones(int(16000 * 0.8), dtype=np.float32)
        chunk2 = np.ones(int(16000 * 0.5), dtype=np.float32) * 2
        
        session.buffer_audio(chunk1)
        result = session.buffer_audio(chunk2)
        
        # total inserted = 1.3s, max = 1.0s -> 16000 samples
        self.assertEqual(len(result), 16000)
        # End should be all 2s
        self.assertEqual(result[-1], 2.0)
        
    def test_session_reset(self):
        session = SessionState("test_4", sample_rate=16000)
        session.buffer_audio(np.ones(16000, dtype=np.float32))
        self.assertEqual(session.audio_duration_seconds, 1.0)
        session.reset()
        self.assertEqual(session.audio_duration_seconds, 0.0)

class TestRuntimeEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RuntimeEngine.from_registry()
        
    def setUp(self):
        if not self.engine.is_ready:
            self.skipTest("No models found in registry, skipping runtime tests.")

    def test_engine_loads_from_registry(self):
        self.assertTrue(self.engine.is_ready)
        self.assertIn("face", self.engine._models)
        self.assertIn("voice", self.engine._models)
        self.assertIn("physio", self.engine._models)

    def test_predict_face_valid(self):
        # Face requires 18 dimensions
        dummy_face = np.random.rand(18)
        res = self.engine.predict_face(dummy_face)
        self.assertNotIn("error", res)
        self.assertIn("stress_probability", res)
        self.assertIn("stress_level", res)
        self.assertTrue(0.0 <= res["stress_probability"] <= 1.0)

    def test_predict_voice_valid(self):
        # Voice requires 12 dimensions
        dummy_voice = np.random.rand(12)
        res = self.engine.predict_voice(dummy_voice)
        self.assertNotIn("error", res)
        self.assertIn("stress_probability", res)
        self.assertIn("stress_level", res)
        self.assertTrue(0.0 <= res["stress_probability"] <= 1.0)

    def test_predict_physio_valid(self):
        # Physio requires 5 dimensions
        dummy_physio = np.random.rand(5)
        res = self.engine.predict_physio(dummy_physio)
        self.assertNotIn("error", res)
        self.assertIn("stress_probability", res)
        self.assertIn("stress_level", res)
        self.assertTrue(0.0 <= res["stress_probability"] <= 1.0)

    def test_predict_fused_three_way(self):
        f18 = np.random.rand(18)
        v12 = np.random.rand(12)
        p5  = np.random.rand(5)
        
        res = self.engine.predict_fused(face=f18, voice=v12, physio=p5)
        self.assertNotIn("error", res)
        self.assertEqual(res["status"], "success")
        self.assertIn("stress_probability", res)
        
        # Check fusion weights (should be 0.3, 0.4, 0.3)
        weights = res["fusion_weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(len(weights), 3)

    def test_predict_fused_graceful_degradation(self):
        """Test fusion dynamically re-weights if a modality is missing."""
        f18 = np.random.rand(18)
        # Voice missing
        res = self.engine.predict_fused(face=f18, physio=np.random.rand(5))
        self.assertNotIn("error", res)
        weights = res["fusion_weights"]
        # Only face and physio should be active
        self.assertEqual(res["active_modalities"], ["face", "physio"])
        self.assertAlmostEqual(sum(w for k, w in weights.items() if k in ["face", "physio"]), 1.0)

    def test_replay_deterministic(self):
        f18 = np.random.rand(18)
        v12 = np.random.rand(12)
        p5  = np.random.rand(5)
        
        rows = [
            {"face": f18, "voice": v12, "physio": p5},
            {"face": f18, "voice": None, "physio": p5},
        ]
        
        # Run replay twice
        results1 = self.engine.replay(rows)
        results2 = self.engine.replay(rows)
        
        # Check determinism
        for r1, r2 in zip(results1, results2):
            self.assertEqual(r1["stress_probability"], r2["stress_probability"])
            self.assertEqual(r1["fusion_weights"], r2["fusion_weights"])

    def test_status_endpoint_data(self):
        status = self.engine.status()
        self.assertEqual(status["engine"], "RuntimeEngine")
        self.assertEqual(status["phase"], 7)
        self.assertTrue(status["ready"])
        self.assertIn("models", status)
        self.assertIn("explainability", status)

if __name__ == "__main__":
    unittest.main()
