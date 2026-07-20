"""
Per-user calibration manager.
Stores personal baseline statistics and applies them to normalize features
before they reach the StandardScaler and classifier.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
import threading
import time

import os

def get_save_path(user_id):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(backend_dir, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir, exist_ok=True)
    return os.path.join(uploads_dir, f"calibration_{user_id}.json")

class UserCalibration:
    """
    Holds calibration data for one user session.
    All values computed during the calm baseline window.
    """
    def __init__(self):
        # Voice calibration
        self.f0_mean       = None   # personal mean pitch Hz
        self.f0_std        = None   # personal pitch variability
        self.rms_mean      = None   # personal voice intensity
        self.rms_std       = None   # personal intensity variability
        self.hnr_mean      = None   # personal HNR baseline
        self.noise_floor   = None   # ambient noise RMS during silence

        # Face calibration
        self.ear_baseline  = None   # personal resting EAR
        self.jaw_baseline  = None   # personal chin-nose/IOD ratio at rest
        self.brow_baseline = None   # personal resting brow descent

        # Physio calibration
        self.physio_mean   = None   # personal mean physiological signals (list)
        self.physio_std    = None   # personal physiological variability (list)

        # Session Scalers
        self.voice_session_scaler = None
        self.face_session_scaler  = None
        self.physio_session_scaler = None
        self._voice_baseline_matrix = []  # rows of 12-dim feature vectors
        self._face_baseline_matrix  = []  # rows of 18-dim indicator vectors
        self._physio_baseline_matrix = [] # rows of 5-dim feature vectors

        # Meta & Verification
        self.is_complete   = False
        self.phase         = 'not_started'
        self.samples_voice = []
        self.samples_face  = []
        self.samples_physio = []
        
        self.is_low_confidence = False
        self.confidence_notes = ""
        self.verification_results = {}
        
        self._lock         = threading.Lock()

    def add_voice_feature_vector(self, feature_vec: np.ndarray):
        """Call this during Phase 2 calibration with raw 12-dim features."""
        with self._lock:
            if feature_vec is not None and len(feature_vec) == 12:
                self._voice_baseline_matrix.append(feature_vec.copy())

    def add_face_feature_vector(self, feature_vec: np.ndarray):
        """Call this during Phase 3 calibration with raw 18-dim features."""
        with self._lock:
            if feature_vec is not None and len(feature_vec) == 18:
                self._face_baseline_matrix.append(feature_vec.copy())

    def add_physio_feature_vector(self, feature_vec: np.ndarray):
        """Call this during calibration with raw 5-dim features."""
        with self._lock:
            if feature_vec is not None and len(feature_vec) == 5:
                self._physio_baseline_matrix.append(feature_vec.copy())

    def build_session_scalers(self):
        """Fit session StandardScaler on the user's own calm baseline data."""
        with self._lock:
            if len(self._voice_baseline_matrix) >= 8:
                X_voice = np.array(self._voice_baseline_matrix)
                self.voice_session_scaler = StandardScaler()
                self.voice_session_scaler.fit(X_voice)

            if len(self._face_baseline_matrix) >= 12:
                X_face = np.array(self._face_baseline_matrix)
                self.face_session_scaler = StandardScaler()
                self.face_session_scaler.fit(X_face)

            if len(self._physio_baseline_matrix) >= 5:
                X_physio = np.array(self._physio_baseline_matrix)
                self.physio_session_scaler = StandardScaler()
                self.physio_session_scaler.fit(X_physio)

            self.is_complete = (
                self.voice_session_scaler is not None and
                self.face_session_scaler  is not None
            )
            return self.is_complete

    def scale_voice_features(self, feature_vec: np.ndarray) -> np.ndarray:
        """Use session scaler to normalize voice features."""
        with self._lock:
            if self.voice_session_scaler is not None:
                return self.voice_session_scaler.transform(feature_vec.reshape(1, -1))
            return None

    def scale_face_features(self, feature_vec: np.ndarray) -> np.ndarray:
        """Use session scaler to normalize face features."""
        with self._lock:
            if self.face_session_scaler is not None:
                return self.face_session_scaler.transform(feature_vec.reshape(1, -1))
            return None

    def scale_physio_features(self, feature_vec: np.ndarray) -> np.ndarray:
        """Use session scaler to normalize physio features."""
        with self._lock:
            if self.physio_session_scaler is not None:
                return self.physio_session_scaler.transform(feature_vec.reshape(1, -1))
            return None

    def add_voice_sample(self, indicators: dict):
        with self._lock:
            self.samples_voice.append(indicators)

    def add_face_sample(self, indicators: dict):
        with self._lock:
            self.samples_face.append(indicators)

    def add_physio_sample(self, indicators: dict):
        with self._lock:
            self.samples_physio.append(indicators)

    def finalize_voice(self):
        """Compute voice baseline statistics from collected samples."""
        with self._lock:
            if len(self.samples_voice) < 5:
                return False

            f0_vals   = [s['f0_mean']        for s in self.samples_voice if s.get('f0_mean', 0) > 60]
            rms_vals  = [s['voice_intensity'] for s in self.samples_voice if s.get('voice_intensity') is not None]
            hnr_vals  = [s['hnr']            for s in self.samples_voice if s.get('hnr') is not None]

            if len(f0_vals) >= 3:
                self.f0_mean  = float(np.median(f0_vals))   # median is more robust than mean
                self.f0_std   = float(np.std(f0_vals)) + 1e-6
            if rms_vals:
                self.rms_mean = float(np.median(rms_vals))
                self.rms_std  = float(np.std(rms_vals)) + 1e-6
            if hnr_vals:
                self.hnr_mean = float(np.median(hnr_vals))

            return True

    def finalize_face(self):
        """Compute face baseline statistics from collected samples."""
        with self._lock:
            if len(self.samples_face) < 10:
                return False

            ear_vals  = [s['avg_ear']           for s in self.samples_face if s.get('avg_ear') is not None]
            jaw_vals  = [s['jaw_displacement']  for s in self.samples_face if s.get('jaw_displacement') is not None]
            brow_vals = [s['brow_descent_left'] for s in self.samples_face if s.get('brow_descent_left') is not None]

            if ear_vals:
                # Remove outliers (blinks) before computing baseline by sorting and slicing the bottom 20%
                ear_arr = np.sort(np.array(ear_vals))
                slice_idx = int(len(ear_arr) * 0.2)
                ear_arr_filtered = ear_arr[slice_idx:]
                if len(ear_arr_filtered) > 0:
                    self.ear_baseline  = float(np.median(ear_arr_filtered))
                else:
                    self.ear_baseline  = float(np.median(ear_arr))

            if jaw_vals:
                self.jaw_baseline  = float(np.median(jaw_vals))

            if brow_vals:
                self.brow_baseline = float(np.median(brow_vals))

            self.is_complete = True
            return True

    def finalize_physio(self):
        """Compute physio baseline statistics from collected samples."""
        with self._lock:
            if len(self.samples_physio) < 5 or len(self._physio_baseline_matrix) < 5:
                return False
            arr = np.array(self._physio_baseline_matrix)
            self.physio_mean = np.median(arr, axis=0).tolist()
            self.physio_std = (np.std(arr, axis=0) + 1e-6).tolist()
            return True

    def normalize_voice_features(self, features: np.ndarray, voice_scaler=None) -> np.ndarray:
        """
        Normalize voice feature vector relative to personal baseline.
        Since features are scaled by voice_scaler.transform() on the server,
        we transform our calibrated user Z-scores back to a pseudo-raw space
        so that when scaled by the StandardScaler, they output the correct Z-score.
        If voice_scaler is None, we return the raw Z-scores directly.
        """
        f = features.copy().astype(np.float32)
        if voice_scaler is None:
            if self.f0_mean is not None and self.f0_mean > 0:
                f[0] = (f[0] - self.f0_mean) / (self.f0_std + 1e-6)
                f[1] = f[1] / (self.f0_std + 1e-6)
                f[2] = f[2] / (self.f0_std + 1e-6)
            if self.rms_mean is not None and self.rms_mean > 0:
                f[7] = (f[7] - self.rms_mean) / (self.rms_std + 1e-6)
            if self.hnr_mean is not None:
                f[5] = f[5] - self.hnr_mean
            return f
            
        means = voice_scaler.mean_
        scales = voice_scaler.scale_

        if self.f0_mean is not None and self.f0_mean > 0:
            # target Z-score for mean F0
            z_f0_mean = (f[0] - self.f0_mean) / (self.f0_std + 1e-6)
            f[0] = means[0] + z_f0_mean * scales[0]
            
            # target Z-score for F0 std
            z_f0_std = f[1] / (self.f0_std + 1e-6)
            f[1] = means[1] + z_f0_std * scales[1]

            # target Z-score for F0 range
            z_f0_range = f[2] / (self.f0_std + 1e-6)
            f[2] = means[2] + z_f0_range * scales[2]

        if self.rms_mean is not None and self.rms_mean > 0:
            # target Z-score for RMS
            z_rms = (f[7] - self.rms_mean) / (self.rms_std + 1e-6)
            f[7] = means[7] + z_rms * scales[7]

        if self.hnr_mean is not None:
            # target Z-score for HNR (deviation)
            z_hnr = f[5] - self.hnr_mean
            f[5] = means[5] + z_hnr * scales[5]

        if self.noise_floor is not None and self.noise_floor > 0:
            # Remove noise floor from high frequency ratio raw feature
            f[8] = max(0.0, f[8] - (self.noise_floor * 2.0))

        return f

    def normalize_face_indicators(self, indicators: dict) -> dict:
        """
        Normalize face indicator dict relative to personal baseline.
        Returns modified copy.
        """
        ind = indicators.copy()

        if self.ear_baseline is not None and self.ear_baseline > 0:
            # EAR relative to personal resting eye openness
            # 0.0 = same as baseline, negative = more closed than baseline (stress)
            raw_ear = ind.get('avg_ear', self.ear_baseline)
            ind['eye_openness_ratio'] = (raw_ear - self.ear_baseline) / (self.ear_baseline + 1e-6)
            # Preserve raw EAR for blink detection but normalize the openness ratio
            ind['avg_ear_normalized'] = ind['eye_openness_ratio']

        if self.jaw_baseline is not None and self.jaw_baseline > 0:
            raw_jaw = ind.get('jaw_displacement', self.jaw_baseline)
            # Negative = jaw more open than baseline, positive = more closed (clenching)
            ind['jaw_displacement_normalized'] = (self.jaw_baseline - raw_jaw) / (self.jaw_baseline + 1e-6)

        if self.brow_baseline is not None and self.brow_baseline > 0:
            raw_brow_l = ind.get('brow_descent_left', self.brow_baseline)
            raw_brow_r = ind.get('brow_descent_right', self.brow_baseline)
            # Positive = brow descended below baseline (furrowing = stress)
            ind['brow_descent_left_normalized']  = (self.brow_baseline - raw_brow_l) / (self.brow_baseline + 1e-6)
            ind['brow_descent_right_normalized'] = (self.brow_baseline - raw_brow_r) / (self.brow_baseline + 1e-6)

        return ind

    def save_to_file(self, user_id):
        import json
        save_path = get_save_path(user_id)
        with self._lock:
            data = {
                'f0_mean': self.f0_mean,
                'f0_std': self.f0_std,
                'rms_mean': self.rms_mean,
                'rms_std': self.rms_std,
                'hnr_mean': self.hnr_mean,
                'noise_floor': self.noise_floor,
                'ear_baseline': self.ear_baseline,
                'jaw_baseline': self.jaw_baseline,
                'brow_baseline': self.brow_baseline,
                'physio_mean': self.physio_mean,
                'physio_std': self.physio_std,
                'is_complete': self.is_complete,
                'is_low_confidence': self.is_low_confidence,
                'confidence_notes': self.confidence_notes,
                'verification_results': self.verification_results,
                'samples_voice': self.samples_voice,
                'samples_face': self.samples_face,
                'samples_physio': self.samples_physio,
                '_voice_baseline_matrix': [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in self._voice_baseline_matrix],
                '_face_baseline_matrix': [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in self._face_baseline_matrix],
                '_physio_baseline_matrix': [row.tolist() if isinstance(row, np.ndarray) else list(row) for row in self._physio_baseline_matrix]
            }
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"[Calibration] Successfully saved calibration to {save_path}")
            return True
        except Exception as e:
            print(f"[Calibration] Error saving calibration to {save_path}: {e}")
            return False

    def load_from_file(self, user_id):
        import json
        save_path = get_save_path(user_id)
        if not os.path.exists(save_path):
            return False
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with self._lock:
                self.f0_mean = data.get('f0_mean')
                self.f0_std = data.get('f0_std')
                self.rms_mean = data.get('rms_mean')
                self.rms_std = data.get('rms_std')
                self.hnr_mean = data.get('hnr_mean')
                self.noise_floor = data.get('noise_floor')
                self.ear_baseline = data.get('ear_baseline')
                self.jaw_baseline = data.get('jaw_baseline')
                self.brow_baseline = data.get('brow_baseline')
                self.physio_mean = data.get('physio_mean')
                self.physio_std = data.get('physio_std')
                self.is_complete = data.get('is_complete', False)
                self.is_low_confidence = data.get('is_low_confidence', False)
                self.confidence_notes = data.get('confidence_notes', "")
                self.verification_results = data.get('verification_results', {})
                self.samples_voice = data.get('samples_voice', [])
                self.samples_face = data.get('samples_face', [])
                self.samples_physio = data.get('samples_physio', [])
                self._voice_baseline_matrix = [np.array(row, dtype=np.float32) for row in data.get('_voice_baseline_matrix', [])]
                self._face_baseline_matrix = [np.array(row, dtype=np.float32) for row in data.get('_face_baseline_matrix', [])]
                self._physio_baseline_matrix = [np.array(row, dtype=np.float32) for row in data.get('_physio_baseline_matrix', [])]
                
                # Reconstruct session scalers
                if len(self._voice_baseline_matrix) >= 8:
                    X_voice = np.array(self._voice_baseline_matrix)
                    self.voice_session_scaler = StandardScaler()
                    self.voice_session_scaler.fit(X_voice)
                if len(self._face_baseline_matrix) >= 12:
                    X_face = np.array(self._face_baseline_matrix)
                    self.face_session_scaler = StandardScaler()
                    self.face_session_scaler.fit(X_face)
                if len(self._physio_baseline_matrix) >= 5:
                    X_physio = np.array(self._physio_baseline_matrix)
                    self.physio_session_scaler = StandardScaler()
                    self.physio_session_scaler.fit(X_physio)
                    
            print(f"[Calibration] Successfully loaded calibration from {save_path}")
            return True
        except Exception as e:
            print(f"[Calibration] Error loading calibration from {save_path}: {e}")
            return False

    def to_dict(self):
        return {
            'f0_mean':       self.f0_mean,
            'f0_std':        self.f0_std,
            'rms_mean':      self.rms_mean,
            'hnr_mean':      self.hnr_mean,
            'ear_baseline':  self.ear_baseline,
            'jaw_baseline':  self.jaw_baseline,
            'brow_baseline': self.brow_baseline,
            'physio_mean':   self.physio_mean,
            'physio_std':    self.physio_std,
            'noise_floor':   self.noise_floor,
            'is_complete':   self.is_complete,
            'is_low_confidence': self.is_low_confidence,
            'confidence_notes': self.confidence_notes,
            'verification_results': self.verification_results,
            'voice_samples': len(self.samples_voice),
            'face_samples':  len(self.samples_face),
            'physio_samples': len(self.samples_physio),
        }


# Session-level store — one calibration object per user_id
_calibrations = {}
_cal_lock = threading.Lock()
_MAX_CALIBRATIONS = 1000

def get_or_create(user_id='default') -> UserCalibration:
    with _cal_lock:
        if user_id not in _calibrations:
            # Evict oldest entry if at capacity to prevent memory leak
            if len(_calibrations) >= _MAX_CALIBRATIONS:
                oldest = next(iter(_calibrations))
                removed = _calibrations.pop(oldest)
                print(f"[Calibration] Evicted oldest calibration for '{oldest}' to maintain max cache size ({_MAX_CALIBRATIONS})")
            _calibrations[user_id] = UserCalibration()
            # Try to load persistent calibration if it exists
            _calibrations[user_id].load_from_file(user_id)
        return _calibrations[user_id]

def clear(user_id='default'):
    with _cal_lock:
        _calibrations.pop(user_id, None)
    # Remove file if exists
    try:
        save_path = get_save_path(user_id)
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"[Calibration] Removed file {save_path}")
        # Also remove the accepted baseline file if exists
        accepted_path = save_path.replace(".json", "_accepted.json")
        if os.path.exists(accepted_path):
            os.remove(accepted_path)
            print(f"[Calibration] Removed accepted file {accepted_path}")
    except Exception as e:
        print(f"[Calibration] Error deleting calibration file: {e}")
