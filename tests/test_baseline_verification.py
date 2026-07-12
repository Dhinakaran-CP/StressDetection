import unittest
import numpy as np
import os
import sys
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT, "backend")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from calibration import UserCalibration, get_or_create, clear, get_save_path
from backend.app import app, runtime_engine

class TestBaselineVerification(unittest.TestCase):
    def setUp(self):
        self.user_id = "test_user_verification"
        clear(self.user_id)
        self.client = app.test_client()

    def tearDown(self):
        clear(self.user_id)
        # Cleanup files if any left
        save_path = get_save_path(self.user_id)
        if os.path.exists(save_path):
            os.remove(save_path)
        accepted_path = save_path.replace(".json", "_accepted.json")
        if os.path.exists(accepted_path):
            os.remove(accepted_path)

    def test_calibration_persistence(self):
        """Verify that calibration can be saved to file and loaded back with reconstructed scalers."""
        cal = get_or_create(self.user_id)
        
        # Add mock voice samples & features
        for i in range(10):
            cal.add_voice_sample({'f0_mean': 120.0 + i, 'voice_intensity': 0.02 + 0.001 * i, 'hnr': 15.0})
            cal.add_voice_feature_vector(np.ones(12, dtype=np.float32) * (1.0 + 0.1 * i))

        # Add mock face samples & features
        for i in range(15):
            cal.add_face_sample({'avg_ear': 0.3, 'jaw_displacement': 0.15, 'brow_descent_left': 0.08})
            cal.add_face_feature_vector(np.ones(18, dtype=np.float32) * (0.5 + 0.05 * i))

        # Add mock physio features
        for i in range(8):
            cal.add_physio_sample({'ecg_rate_mean': 70.0})
            cal.add_physio_feature_vector(np.ones(5, dtype=np.float32) * (2.0 + i))

        # Finalize and build scalers
        voice_ok = cal.finalize_voice()
        face_ok = cal.finalize_face()
        phys_ok = cal.finalize_physio()
        scalers_built = cal.build_session_scalers()

        self.assertTrue(voice_ok)
        self.assertTrue(face_ok)
        self.assertTrue(phys_ok)
        self.assertTrue(scalers_built)
        self.assertTrue(cal.is_complete)

        # Save to file
        saved = cal.save_to_file(self.user_id)
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(get_save_path(self.user_id)))

        # Clear in-memory cache without deleting file on disk
        from calibration import _calibrations, _cal_lock
        with _cal_lock:
            _calibrations.pop(self.user_id, None)
        
        # Load back
        cal_loaded = get_or_create(self.user_id)
        self.assertTrue(cal_loaded.is_complete)
        self.assertEqual(cal_loaded.f0_mean, cal.f0_mean)
        self.assertEqual(cal_loaded.ear_baseline, cal.ear_baseline)
        self.assertEqual(cal_loaded.physio_mean, cal.physio_mean)
        self.assertIsNotNone(cal_loaded.voice_session_scaler)
        self.assertIsNotNone(cal_loaded.face_session_scaler)
        self.assertIsNotNone(cal_loaded.physio_session_scaler)

    def test_finalize_verification_and_simulated_physio(self):
        """Test finalization logic, simulated physio fallback, and uncalibrated verification."""
        cal = get_or_create(self.user_id)
        
        # Collect voice & face samples (but NO physio samples to trigger simulation)
        for i in range(10):
            cal.add_voice_sample({'f0_mean': 120.0 + i, 'voice_intensity': 0.02 + 0.001 * i, 'hnr': 15.0})
            cal.add_voice_feature_vector(np.ones(12, dtype=np.float32) * 1.5)

        for i in range(15):
            cal.add_face_sample({'avg_ear': 0.3, 'jaw_displacement': 0.15, 'brow_descent_left': 0.08})
            cal.add_face_feature_vector(np.ones(18, dtype=np.float32) * 0.8)

        # Finalize through API
        response = self.client.post('/api/calibrate/finalize', json={'user_id': self.user_id})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        self.assertEqual(data['status'], 'complete')
        self.assertTrue(data['session_scalers_built'])
        self.assertIn('verification', data)
        
        # Verify simulated physio worked
        cal_check = get_or_create(self.user_id)
        self.assertIsNotNone(cal_check.physio_mean)
        self.assertEqual(len(cal_check.samples_physio), 10)
        self.assertEqual(len(cal_check._physio_baseline_matrix), 10)
        
        # Verify verification results are present
        ver = data['verification']
        self.assertIn('recommendation', ver)
        self.assertIn('stress_probability', ver)
        self.assertIn('biomarker_scores', ver)
        self.assertIn('pop_deviations', ver)
        self.assertIn('explanation_summary', ver)

    def test_consent_loop_states(self):
        """Test accept_low_confidence, recalibrate, and discard state transitions."""
        # 1. Setup a calibration session
        cal = get_or_create(self.user_id)
        for i in range(10):
            cal.add_voice_sample({'f0_mean': 120.0, 'voice_intensity': 0.02, 'hnr': 15.0})
            cal.add_voice_feature_vector(np.ones(12, dtype=np.float32))
        for i in range(15):
            cal.add_face_sample({'avg_ear': 0.3, 'jaw_displacement': 0.15, 'brow_descent_left': 0.08})
            cal.add_face_feature_vector(np.ones(18, dtype=np.float32))

        # Finalize first to calculate baseline
        self.client.post('/api/calibrate/finalize', json={'user_id': self.user_id})

        # 2. Test manual confirmation (low confidence)
        response = self.client.post('/api/calibrate/confirm', json={
            'user_id': self.user_id,
            'action': 'accept_low_confidence',
            'notes': 'Test validation note'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['state'], 'ACCEPT_WITH_LOW_CONFIDENCE')
        self.assertTrue(data['calibration']['is_low_confidence'])
        self.assertEqual(data['calibration']['confidence_notes'], 'Test validation note')
        
        # Verify accepted file was created
        accepted_path = get_save_path(self.user_id).replace(".json", "_accepted.json")
        self.assertTrue(os.path.exists(accepted_path))

        # 3. Test recalibrate/discard cleans files
        response = self.client.post('/api/calibrate/confirm', json={
            'user_id': self.user_id,
            'action': 'recalibrate'
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['state'], 'RECALIBRATE')
        self.assertFalse(os.path.exists(get_save_path(self.user_id)))
        self.assertFalse(os.path.exists(accepted_path))

if __name__ == '__main__':
    unittest.main()
