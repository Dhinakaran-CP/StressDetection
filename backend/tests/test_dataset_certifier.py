import unittest
import pandas as pd
from backend.core.dataset_certifier import DatasetCertifier

class TestDatasetCertifier(unittest.TestCase):
    def setUp(self):
        self.certifier = DatasetCertifier("contracts/schema_contract.yaml")
        
        # Valid minimal DataFrame based on schema
        self.valid_df = pd.DataFrame({
            "subject_id": ["subject01", "subject01"],
            "task_id": ["math", "math"],
            "video_id": ["vid1", "vid1"],
            "window_index": [0, 1],
            "window_start": [0.0, 0.5],
            "window_end": [1.0, 1.5],
            "label": [1, 1],
            "fake_feature": [0.1, 0.2]
        })

    def test_valid_dataframe(self):
        report = self.certifier.validate(self.valid_df, "Test")
        self.assertEqual(report["status"], "CERTIFIED")
        self.assertEqual(report["total_rows"], 2)

    def test_missing_required_column(self):
        bad_df = self.valid_df.drop(columns=["subject_id"])
        with self.assertRaises(ValueError) as context:
            self.certifier.validate(bad_df, "Bad")
        self.assertIn("Missing required columns", str(context.exception))

    def test_missing_subject_id_values(self):
        bad_df = self.valid_df.copy()
        bad_df.loc[0, "subject_id"] = None
        with self.assertRaises(ValueError) as context:
            self.certifier.validate(bad_df, "Bad")
        self.assertIn("missing subject_id", str(context.exception))

    def test_duplicate_keys(self):
        bad_df = self.valid_df.copy()
        bad_df.loc[1, "window_index"] = 0 # Duplicate key (subject01, math, 0)
        with self.assertRaises(ValueError) as context:
            self.certifier.validate(bad_df, "Bad")
        self.assertIn("duplicate rows", str(context.exception))

    def test_non_monotonic_time(self):
        bad_df = self.valid_df.copy()
        bad_df.loc[1, "window_start"] = -1.0 # Goes backwards in time
        with self.assertRaises(ValueError) as context:
            self.certifier.validate(bad_df, "Bad")
        self.assertIn("is not monotonic", str(context.exception))

if __name__ == '__main__':
    unittest.main()
