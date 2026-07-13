import numpy as np

class SignalQualityMonitor:
    """
    Evaluates real-time signal quality for Face, Voice, and Physio streams
    based on feature ranges and sensor telemetry boundaries.
    """
    def __init__(self, face_min_confidence=0.5, voice_min_intensity=0.01, physio_hr_bounds=(40.0, 180.0)):
        self.face_min_confidence = face_min_confidence
        self.voice_min_intensity = voice_min_intensity
        self.physio_hr_min, self.physio_hr_max = physio_hr_bounds

    def evaluate_face_quality(self, features):
        """
        Expects a numpy array of shape (18,) corresponding to FACE_FEATURES.
        Returns a float score in [0.0, 1.0].
        """
        if features is None or len(features) < 18:
            return 0.0
        
        # landmark_confidence is at index 16
        confidence = features[16]
        if np.isnan(confidence) or confidence < self.face_min_confidence:
            return 0.0
            
        # Return normalized confidence
        return float(np.clip(confidence, 0.0, 1.0))

    def evaluate_voice_quality(self, features):
        """
        Expects a numpy array of shape (12,) corresponding to VOICE_FEATURES.
        Returns a float score in [0.0, 1.0].
        """
        if features is None or len(features) < 12:
            return 0.0
            
        # voice_intensity is at index 7, hnr is at index 5
        intensity = features[7]
        hnr = features[5]
        
        if np.isnan(intensity) or intensity < self.voice_min_intensity:
            return 0.0
            
        # Standard vocal quality score combining intensity and harmonics ratio
        score = 0.5 * (np.clip(intensity * 10, 0.0, 1.0)) + 0.5 * (np.clip(hnr / 30.0, 0.0, 1.0))
        return float(np.nan_to_num(score, nan=0.0))

    def evaluate_physio_quality(self, features):
        """
        Expects a numpy array of shape (5,) corresponding to PHYSIO_FEATURES.
        Returns a float score in [0.0, 1.0].
        """
        if features is None or len(features) < 5:
            return 0.0
            
        hr_mean = features[0]       # ecg_rate_mean
        hrv_rmssd = features[1]     # ecg_hrv_rmssd
        eda_scl = features[3]       # eda_scl_mean
        
        # Check physiological extremes indicating sensor dislocation/noise
        if np.isnan(hr_mean) or hr_mean < self.physio_hr_min or hr_mean > self.physio_hr_max:
            return 0.0
        if np.isnan(hrv_rmssd) or hrv_rmssd < 0.0 or hrv_rmssd > 500.0:
            return 0.0
        if np.isnan(eda_scl) or eda_scl <= 0.0:
            return 0.0
            
        return 1.0

    def get_quality_status(self, modality, features):
        """
        Evaluates the quality of a specific modality stream.
        """
        if modality == "face":
            score = self.evaluate_face_quality(features)
        elif modality == "voice":
            score = self.evaluate_voice_quality(features)
        elif modality == "physio":
            score = self.evaluate_physio_quality(features)
        else:
            raise ValueError(f"Unknown modality: {modality}")
            
        return {
            "modality": modality,
            "quality_score": score,
            "is_reliable": score > 0.0
        }
