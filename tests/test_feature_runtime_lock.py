import unittest
import numpy as np
from backend.core.feature_runtime_lock import FeatureRuntimeLock

class DummyScaler:
    def transform(self, X):
        return X * 2.0

class TestFeatureRuntimeLock(unittest.TestCase):
    def setUp(self):
        self.lock = FeatureRuntimeLock("configs/feature_contract.yaml")
        self.scaler = DummyScaler()
        
    def test_face_features_valid(self):
        raw = np.ones(18)
        processed = self.lock.process_face_features(raw, self.scaler)
        self.assertEqual(processed.shape, (1, 18))
        self.assertEqual(processed[0][0], 2.0) # Scaler was applied
        
    def test_face_features_invalid_dim(self):
        raw = np.ones(10) # Contract says 18
        with self.assertRaises(ValueError) as context:
            self.lock.process_face_features(raw, self.scaler)
        self.assertIn("expected 18 dims, got 10", str(context.exception))
        
    def test_face_features_none_handling(self):
        processed = self.lock.process_face_features(None, None)
        self.assertEqual(processed.shape, (1, 18))
        self.assertEqual(processed[0][0], 0.0)

    def test_voice_features_valid(self):
        raw = np.ones(12)
        processed = self.lock.process_voice_features(raw, self.scaler)
        self.assertEqual(processed.shape, (1, 12))
        self.assertEqual(processed[0][0], 2.0)
        
    def test_voice_features_invalid_dim(self):
        raw = np.ones(15) # Contract says 12
        with self.assertRaises(ValueError) as context:
            self.lock.process_voice_features(raw, self.scaler)
        self.assertIn("expected 12 dims, got 15", str(context.exception))

if __name__ == '__main__':
    unittest.main()
