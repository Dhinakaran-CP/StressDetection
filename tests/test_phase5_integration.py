"""
test_phase5_integration.py
Comprehensive integration test covering Phases 1-5:
  - FeatureRuntimeLock (face, voice, physio)
  - DatasetCertifier schema validation
  - Weighted fusion logic (model.py fuse_predictions)
  - End-to-end model.predict() with 3 modalities
  - Version Registry
"""
import os, sys, unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -------------------------------------------------------
# 1. FeatureRuntimeLock
# -------------------------------------------------------
from backend.core.feature_runtime_lock import FeatureRuntimeLock

class DummyScaler:
    def transform(self, X): return X * 2.0

class TestFeatureRuntimeLock(unittest.TestCase):
    def setUp(self):
        self.lock = FeatureRuntimeLock("configs/feature_contract.yaml")
        self.scaler = DummyScaler()

    def test_face_valid(self):
        out = self.lock.process_face_features(np.ones(18), self.scaler)
        self.assertEqual(out.shape, (1, 18))
        self.assertAlmostEqual(out[0][0], 2.0)

    def test_face_none_zeros(self):
        out = self.lock.process_face_features(None, None)
        self.assertEqual(out.shape, (1, 18))
        self.assertEqual(out[0][0], 0.0)

    def test_face_wrong_dim(self):
        with self.assertRaises(ValueError):
            self.lock.process_face_features(np.ones(10), None)

    def test_face_nan_filled(self):
        out = self.lock.process_face_features(np.array([float("nan")] * 18), None)
        self.assertEqual(out[0][0], 0.0)

    def test_voice_valid(self):
        out = self.lock.process_voice_features(np.ones(12), self.scaler)
        self.assertEqual(out.shape, (1, 12))
        self.assertAlmostEqual(out[0][0], 2.0)

    def test_voice_wrong_dim(self):
        with self.assertRaises(ValueError):
            self.lock.process_voice_features(np.ones(5), None)

    def test_physio_valid(self):
        out = self.lock.process_physio_features(np.ones(5), self.scaler)
        self.assertEqual(out.shape, (1, 5))
        self.assertAlmostEqual(out[0][0], 2.0)

    def test_physio_none_zeros(self):
        out = self.lock.process_physio_features(None, None)
        self.assertEqual(out.shape, (1, 5))
        self.assertEqual(out[0][0], 0.0)

    def test_physio_wrong_dim(self):
        with self.assertRaises(ValueError):
            self.lock.process_physio_features(np.ones(51), None)

    def test_physio_nan_filled(self):
        out = self.lock.process_physio_features(np.array([float("nan")] * 5), None)
        self.assertEqual(out[0][0], 0.0)

# -------------------------------------------------------
# 2. DatasetCertifier
# -------------------------------------------------------
from backend.core.dataset_certifier import DatasetCertifier

def _make_valid_df(n=2, w=3):
    rows = []
    for s in range(n):
        for i in range(w):
            rows.append({"subject_id": f"s{s:02d}", "task_id": "math",
                         "video_id": "v1", "window_index": i,
                         "window_start": i*0.5, "window_end": i*0.5+1.0,
                         "label": 0, "feat": 0.1})
    return pd.DataFrame(rows)

class TestDatasetCertifier(unittest.TestCase):
    def setUp(self):
        self.cert = DatasetCertifier("configs/schema_contract.yaml")

    def test_valid_passes(self):
        r = self.cert.validate(_make_valid_df(), "Test")
        self.assertEqual(r["status"], "CERTIFIED")

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.cert.validate(_make_valid_df().drop(columns=["subject_id"]), "X")
        self.assertIn("Missing required columns", str(ctx.exception))

    def test_null_subject_id_raises(self):
        df = _make_valid_df(); df.loc[0, "subject_id"] = None
        with self.assertRaises(ValueError) as ctx:
            self.cert.validate(df, "X")
        self.assertIn("missing subject_id", str(ctx.exception))

    def test_duplicate_key_raises(self):
        df = _make_valid_df(); df.loc[1, "window_index"] = 0
        with self.assertRaises(ValueError) as ctx:
            self.cert.validate(df, "X")
        self.assertIn("duplicate rows", str(ctx.exception))

    def test_non_monotonic_raises(self):
        df = _make_valid_df(); df.loc[1, "window_start"] = -1.0
        with self.assertRaises(ValueError) as ctx:
            self.cert.validate(df, "X")
        self.assertIn("monotonic", str(ctx.exception))

    def test_certified_csvs_exist(self):
        for m in ("face", "voice", "physio"):
            p = f"certified_data/{m}_certified.csv"
            if os.path.exists(p):
                df = pd.read_csv(p, nrows=5)
                self.assertIn("subject_id", df.columns)
                self.assertIn("label", df.columns)

# -------------------------------------------------------
# 3. Weighted Fusion (model.py)
# -------------------------------------------------------
from backend.model import fuse_predictions

class TestFusePredictions(unittest.TestCase):

    def test_empty(self):
        r = fuse_predictions({}, {})
        self.assertEqual(r["fused_score"], 0.0)
        self.assertEqual(r["stress_level"], "Low")

    def test_single_passthrough(self):
        r = fuse_predictions({"voice": 0.8}, {})
        self.assertAlmostEqual(r["fused_score"], 0.8)
        self.assertEqual(r["stress_level"], "High")

    def test_two_way_renormalised(self):
        r = fuse_predictions({"face": 0.6, "voice": 0.7}, {})
        expected = (0.6*0.30 + 0.7*0.40) / (0.30 + 0.40)
        self.assertAlmostEqual(r["fused_score"], expected, places=5)

    def test_three_way_phase4_weights(self):
        r = fuse_predictions({"face": 0.6, "voice": 0.7, "physio": 0.5}, {})
        expected = 0.6*0.30 + 0.7*0.40 + 0.5*0.30
        self.assertAlmostEqual(r["fused_score"], expected, places=5)

    def test_high_threshold(self):
        self.assertEqual(fuse_predictions({"voice": 0.8}, {})["stress_level"], "High")

    def test_moderate_threshold(self):
        self.assertEqual(fuse_predictions({"voice": 0.5}, {})["stress_level"], "Moderate")

    def test_low_threshold(self):
        self.assertEqual(fuse_predictions({"voice": 0.2}, {})["stress_level"], "Low")

    def test_unknown_modality_ignored(self):
        r = fuse_predictions({"eeg": 0.9}, {})
        self.assertEqual(r["fused_score"], 0.0)

# -------------------------------------------------------
# 4. End-to-end predict()
# -------------------------------------------------------
from backend.model import MultimodalStressDetector

class TestMultimodalPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = MultimodalStressDetector()
        cls.loaded = cls.det.load_model(ROOT)

    def test_models_loaded(self):
        self.assertTrue(self.loaded)

    def test_no_features_errors(self):
        self.assertIn("error", self.det.predict())

    def test_voice_only_success(self):
        r = self.det.predict(voice_features=np.random.rand(12))
        self.assertEqual(r.get("status"), "success")

    def test_three_way_fields_present(self):
        r = self.det.predict(
            facial_features=np.random.rand(18),
            voice_features=np.random.rand(12),
            phys_features=np.random.rand(5),
        )
        self.assertEqual(r.get("status"), "success")
        for field in ("stress_probability", "stress_level", "fusion_weights", "individual_predictions"):
            self.assertIn(field, r)

    def test_fusion_weights_sum_to_one(self):
        r = self.det.predict(
            facial_features=np.random.rand(18),
            voice_features=np.random.rand(12),
            phys_features=np.random.rand(5),
        )
        w = {k: v for k, v in r.get("fusion_weights", {}).items() if v and v > 0}
        if w:
            self.assertAlmostEqual(sum(w.values()), 1.0, places=3)

    def test_stress_prob_in_range(self):
        r = self.det.predict(voice_features=np.random.rand(12))
        p = r.get("stress_probability", -1)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_phase4_weights_when_all_active(self):
        r = self.det.predict(
            facial_features=np.random.rand(18),
            voice_features=np.random.rand(12),
            phys_features=np.random.rand(5),
        )
        w = r.get("fusion_weights", {})
        if self.det.phys_model:
            self.assertAlmostEqual(w.get("facial", 0), 0.3, places=2)
            self.assertAlmostEqual(w.get("voice", 0), 0.4, places=2)
            self.assertAlmostEqual(w.get("physiological", 0), 0.3, places=2)

# -------------------------------------------------------
# 5. Version Registry
# -------------------------------------------------------
from backend.core.version_registry import VersionRegistry

class TestVersionRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = VersionRegistry()

    def test_loads(self):
        self.assertIsNotNone(self.reg)

    def test_face_expert_has_accuracy(self):
        m = self.reg.get_active_model("face_expert")
        if m:
            self.assertIn("accuracy", m["metadata"])

    def test_physio_expert_registered(self):
        m = self.reg.get_active_model("physio_expert")
        if m:
            self.assertIn("accuracy", m["metadata"])
            self.assertGreater(m["metadata"]["accuracy"], 0.5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
