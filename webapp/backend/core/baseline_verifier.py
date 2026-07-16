import numpy as np

class BaselineVerifier:
    """
    Validates calibration baseline sequences to ensure the user is in a calm,
    low-stress, and clean sensor state during calibration.
    """
    def __init__(self, calm_hr_threshold=95.0, face_min_confidence=0.8, voice_max_intensity=0.08):
        self.calm_hr_threshold = calm_hr_threshold
        self.face_min_confidence = face_min_confidence
        self.voice_max_intensity = voice_max_intensity

    def verify_face_baseline(self, features_chunk):
        """
        Validates a sequence of face features (shape [N, 18]).
        Checks if tracking was stable and user was relatively still.
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "recalibration_needed", "reason": "No data received"}

        # confidence is at index 16
        confidences = features_chunk[:, 16]
        mean_conf = np.mean(confidences)
        if mean_conf < self.face_min_confidence:
            return {
                "status": "recalibration_needed",
                "reason": f"Face tracking confidence too low: {mean_conf:.2f} (expected > {self.face_min_confidence})"
            }

        # head_tilt is at index 12, temporal_x_var at 13
        # Check for excessive movement during calibration
        movement = np.std(features_chunk[:, 12])
        if movement > 0.15:
            return {
                "status": "low_confidence",
                "reason": f"Excessive head movement standard deviation: {movement:.3f} (expected < 0.15)"
            }

        return {"status": "verified", "reason": "Stable face baseline verified"}

    def verify_voice_baseline(self, features_chunk):
        """
        Validates voice sequence (shape [N, 12]).
        Ensures the baseline calibration is mostly silent.
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "verified", "reason": "No voice data (optional)"}

        # voice_intensity is at index 7
        intensities = features_chunk[:, 7]
        mean_intensity = np.mean(intensities)
        
        if mean_intensity > self.voice_max_intensity:
            return {
                "status": "recalibration_needed",
                "reason": f"Vocal intensity too high: {mean_intensity:.3f} (expected silence/low noise < {self.voice_max_intensity})"
            }

        return {"status": "verified", "reason": "Voice baseline is suitably quiet"}

    def verify_physio_baseline(self, features_chunk):
        """
        Validates physiological sequence (shape [N, 5]).
        Ensures the user's heart rate is in a normal resting state (not elevated).
        """
        if features_chunk is None or len(features_chunk) == 0:
            return {"status": "recalibration_needed", "reason": "No physiological data received"}

        # hr_mean is at index 0
        hrs = features_chunk[:, 0]
        mean_hr = np.mean(hrs)

        if mean_hr > self.calm_hr_threshold:
            return {
                "status": "recalibration_needed",
                "reason": f"Elevated heart rate detected: {mean_hr:.1f} bpm (expected calm resting HR < {self.calm_hr_threshold})"
            }

        return {"status": "verified", "reason": "Physiological baseline verified in resting range"}

    def verify_calibration_session(self, face_data=None, voice_data=None, physio_data=None):
        """
        Aggregates results for all modalities to grant or deny baseline verification status.
        """
        reports = {}
        if face_data is not None:
            reports["face"] = self.verify_face_baseline(face_data)
        if voice_data is not None:
            reports["voice"] = self.verify_voice_baseline(voice_data)
        if physio_data is not None:
            reports["physio"] = self.verify_physio_baseline(physio_data)

        # Determine overall status
        overall_status = "verified"
        reasons = []
        
        for mod, r in reports.items():
            if r["status"] == "recalibration_needed":
                overall_status = "recalibration_needed"
                reasons.append(f"{mod}: {r['reason']}")
            elif r["status"] == "low_confidence" and overall_status != "recalibration_needed":
                overall_status = "low_confidence"
                reasons.append(f"{mod}: {r['reason']}")

        return {
            "status": overall_status,
            "reports": reports,
            "summary_reason": "; ".join(reasons) if reasons else "Calibration verified successfully"
        }
