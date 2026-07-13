"""
test_explainability_bundle.py
Phase 6 test suite for the Explainability Release Pipeline.
Tests the contract, bundle loader, and engine API.
"""
import os
import sys
import json
import tempfile
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.explainability.explainability_contract import (
    FACE_FEATURE_LABELS, VOICE_FEATURE_LABELS, PHYSIO_FEATURE_LABELS,
    MODALITY_LABELS, MODALITY_GROUPS,
    REQUIRED_BUNDLE_KEYS, REQUIRED_MODEL_KEYS,
    TOP_K_PER_MODALITY, TOP_K_GLOBAL,
    BUNDLE_VERSION,
)
from backend.explainability.explainability_engine import ExplainabilityEngine
from backend.core.version_registry import VersionRegistry

BUNDLE_PATH = os.path.join(ROOT, "models", "explainability_bundle.json")

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def _make_fake_bundle(modalities=("face", "voice", "physio")):
    """Build a minimal valid bundle dict for engine unit tests."""
    models = {}
    for mod in modalities:
        labels = MODALITY_LABELS.get(mod, [])
        groups = MODALITY_GROUPS.get(mod, [])
        n = len(labels)
        top_k = [
            {
                "feature_index": i,
                "feature_label": labels[i] if i < len(labels) else f"{mod}_{i}",
                "feature_group": groups[i] if i < len(groups) else "unknown",
                "mean_abs_shap": float(n - i),
            }
            for i in range(min(TOP_K_PER_MODALITY, n))
        ]
        models[mod] = {
            "model_file":     f"{mod}_expert_lightweight.pkl",
            "n_features":     n,
            "sample_size":    100,
            "feature_labels": labels,
            "feature_groups": groups,
            "shap_means":     [float(n - i) for i in range(n)],
            "top_features":   top_k,
            "shap_available": True,
        }
    return {
        "version":        BUNDLE_VERSION,
        "created_at":     "2026-07-01T00:00:00Z",
        "shap_available": True,
        "models":         models,
    }


def _write_fake_bundle(bundle_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle_dict, f)


# -------------------------------------------------------
# 1. Contract — label counts and alignment
# -------------------------------------------------------
class TestExplainabilityContract(unittest.TestCase):

    def test_face_label_count(self):
        self.assertEqual(len(FACE_FEATURE_LABELS), 18)

    def test_voice_label_count(self):
        self.assertEqual(len(VOICE_FEATURE_LABELS), 12)

    def test_physio_label_count(self):
        self.assertEqual(len(PHYSIO_FEATURE_LABELS), 5)

    def test_modality_labels_complete(self):
        for mod in ("face", "voice", "physio"):
            self.assertIn(mod, MODALITY_LABELS)
            self.assertGreater(len(MODALITY_LABELS[mod]), 0)

    def test_modality_groups_aligned(self):
        for mod in ("face", "voice", "physio"):
            self.assertEqual(
                len(MODALITY_LABELS[mod]),
                len(MODALITY_GROUPS[mod]),
                f"Groups and labels must have same length for {mod}",
            )

    def test_no_duplicate_labels(self):
        for mod, labels in MODALITY_LABELS.items():
            self.assertEqual(len(labels), len(set(labels)),
                             f"Duplicate labels found in {mod}")

    def test_top_k_constants(self):
        self.assertGreater(TOP_K_PER_MODALITY, 0)
        self.assertGreater(TOP_K_GLOBAL, 0)
        self.assertGreaterEqual(TOP_K_GLOBAL, TOP_K_PER_MODALITY)


# -------------------------------------------------------
# 2. Bundle file — if it was built, verify its contents
# -------------------------------------------------------
class TestBundleFile(unittest.TestCase):

    def setUp(self):
        self.bundle_exists = os.path.exists(BUNDLE_PATH)
        if self.bundle_exists:
            with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
                self.bundle = json.load(f)
        else:
            self.bundle = None

    def test_bundle_file_exists(self):
        self.assertTrue(
            self.bundle_exists,
            f"explainability_bundle.json not found at {BUNDLE_PATH}. "
            "Run: python -m backend.explainability.build_explainability_bundle"
        )

    def test_bundle_has_required_keys(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for key in REQUIRED_BUNDLE_KEYS:
            self.assertIn(key, self.bundle)

    def test_bundle_has_at_least_one_modality(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        self.assertGreater(len(self.bundle["models"]), 0)

    def test_each_modality_has_required_keys(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            for key in REQUIRED_MODEL_KEYS:
                self.assertIn(key, data, f"Key '{key}' missing for modality '{mod}'")

    def test_feature_labels_match_contract(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            contract_labels = MODALITY_LABELS.get(mod, [])
            bundle_labels   = data.get("feature_labels", [])
            n = min(len(contract_labels), len(bundle_labels))
            for i in range(n):
                self.assertEqual(
                    contract_labels[i], bundle_labels[i],
                    f"Label mismatch at index {i} for {mod}"
                )

    def test_shap_means_correct_length(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            expected = data.get("n_features", 0)
            actual   = len(data.get("shap_means", []))
            self.assertEqual(actual, expected,
                             f"shap_means length mismatch for {mod}")

    def test_top_features_count(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            count = len(data.get("top_features", []))
            self.assertGreater(count, 0, f"No top_features for {mod}")
            self.assertLessEqual(count, TOP_K_PER_MODALITY + 1)

    def test_top_feature_has_required_fields(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            for feat in data.get("top_features", []):
                self.assertIn("feature_index", feat)
                self.assertIn("feature_label", feat)
                self.assertIn("mean_abs_shap", feat)

    def test_shap_means_are_non_negative(self):
        if not self.bundle_exists: self.skipTest("Bundle not built yet")
        for mod, data in self.bundle["models"].items():
            for val in data.get("shap_means", []):
                self.assertGreaterEqual(float(val), 0.0,
                    f"Negative mean_abs_shap for {mod}")


# -------------------------------------------------------
# 3. ExplainabilityEngine — unit tests with fake bundle
# -------------------------------------------------------
class TestExplainabilityEngine(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bundle_path = os.path.join(self.tmpdir, "explainability_bundle.json")
        bundle = _make_fake_bundle()
        _write_fake_bundle(bundle, self.bundle_path)
        self.engine = ExplainabilityEngine(bundle_path=self.bundle_path)

    def test_engine_loads(self):
        self.assertTrue(self.engine.is_loaded)

    def test_status_returns_version(self):
        s = self.engine.status()
        self.assertTrue(s["loaded"])
        self.assertEqual(s["version"], BUNDLE_VERSION)

    def test_explain_face_returns_top_k(self):
        raw = np.random.rand(18)
        result = self.engine.explain_modality("face", raw)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["top_features"]), 0)
        self.assertLessEqual(len(result["top_features"]), TOP_K_PER_MODALITY)

    def test_explain_voice_returns_top_k(self):
        result = self.engine.explain_modality("voice", np.random.rand(12))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["top_features"]), 0)

    def test_explain_physio_returns_top_k(self):
        result = self.engine.explain_modality("physio", np.random.rand(5))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["top_features"]), 0)

    def test_explain_unknown_modality_unavailable(self):
        result = self.engine.explain_modality("eeg", None)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["top_features"], [])

    def test_feature_values_injected(self):
        raw = np.array([0.42] + [0.0] * 17)
        result = self.engine.explain_modality("face", raw)
        # Feature 0 should have value 0.42
        feat0 = next((f for f in result["top_features"] if f["feature_index"] == 0), None)
        if feat0 is not None:
            self.assertAlmostEqual(feat0["feature_value"], 0.42, places=2)

    def test_full_payload_contains_all_modalities(self):
        payload = self.engine.build_full_payload(
            face_features=np.random.rand(18),
            voice_features=np.random.rand(12),
            physio_features=np.random.rand(5),
        )
        modality_names = [m["modality"] for m in payload["modalities"]]
        self.assertIn("face", modality_names)
        self.assertIn("voice", modality_names)
        self.assertIn("physio", modality_names)

    def test_global_top_drivers_capped(self):
        payload = self.engine.build_full_payload(
            face_features=np.random.rand(18),
            voice_features=np.random.rand(12),
            physio_features=np.random.rand(5),
        )
        self.assertLessEqual(len(payload["top_drivers"]), TOP_K_GLOBAL)

    def test_global_top_drivers_sorted_desc(self):
        payload = self.engine.build_full_payload(
            face_features=np.random.rand(18),
            voice_features=np.random.rand(12),
        )
        drivers = payload["top_drivers"]
        for i in range(len(drivers) - 1):
            self.assertGreaterEqual(
                abs(drivers[i]["mean_abs_shap"]),
                abs(drivers[i + 1]["mean_abs_shap"]),
            )


# -------------------------------------------------------
# 4. ExplainabilityEngine — graceful fallback when no bundle
# -------------------------------------------------------
class TestEngineNoBundle(unittest.TestCase):

    def setUp(self):
        self.engine = ExplainabilityEngine(bundle_path="/nonexistent/path.json")

    def test_engine_not_loaded(self):
        self.assertFalse(self.engine.is_loaded)

    def test_status_reports_error(self):
        s = self.engine.status()
        self.assertFalse(s["loaded"])
        self.assertIsNotNone(s["error"])

    def test_explain_returns_unavailable(self):
        result = self.engine.explain_modality("face")
        self.assertEqual(result["status"], "unavailable")

    def test_full_payload_available_false(self):
        payload = self.engine.build_full_payload()
        self.assertFalse(payload["available"])
        self.assertIsNotNone(payload["message"])


# -------------------------------------------------------
# 5 (addition). VersionRegistry — bundle registration
# -------------------------------------------------------
class TestVersionRegistryBundle(unittest.TestCase):
    def setUp(self):
        self.reg = VersionRegistry()

    def test_active_bundles_key_exists(self):
        """registry.json must contain active_bundles after Phase 6."""
        self.assertIn("active_bundles", self.reg._registry)

    def test_explainability_bundle_registered(self):
        bundle_info = self.reg.get_active_bundle("explainability_bundle")
        if bundle_info:
            self.assertEqual(bundle_info.get("artifact_type"), "bundle")
            self.assertIn("hash", bundle_info)
            self.assertIsNotNone(bundle_info["hash"])
            meta = bundle_info.get("metadata", {})
            self.assertIn("modalities", meta)
            self.assertIn("top_drivers", meta)

    def test_bundle_hash_matches_file(self):
        import hashlib
        bundle_info = self.reg.get_active_bundle("explainability_bundle")
        if not bundle_info or not bundle_info.get("hash"):
            self.skipTest("Bundle not yet registered")
        bundle_path = os.path.join(ROOT, "models", "explainability_bundle.json")
        sha256 = hashlib.sha256()
        with open(bundle_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        self.assertEqual(sha256.hexdigest(), bundle_info["hash"])

if __name__ == "__main__":
    unittest.main(verbosity=2)

