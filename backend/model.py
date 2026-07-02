"""
Model.py — Phase 5 (Engine Integration & Explainability)
Extraction logic lives in backend/core/extractors/.
Transformations are governed by backend/core/feature_runtime_lock.py.
Fusion uses validated Phase 4 weights: Face=0.30, Voice=0.40, Physio=0.30.
"""
import os
import numpy as np
import cv2
import pickle
import sys

from backend.core.extractors.face_extractor import FaceExtractor
from backend.core.extractors.voice_extractor import VoiceExtractor
from backend.core.feature_runtime_lock import FeatureRuntimeLock

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'sklearn._loss' in module:
            try:
                return super().find_class(module, name)
            except (ImportError, ModuleNotFoundError):
                for alt_module in ['sklearn._loss', 'sklearn._loss._loss', 'sklearn._loss.loss']:
                    try:
                        __import__(alt_module)
                        m = sys.modules[alt_module]
                        if hasattr(m, name):
                            return getattr(m, name)
                    except (ImportError, KeyError, AttributeError):
                        continue
        return super().find_class(module, name)

def safe_pickle_load(file_obj):
    return CustomUnpickler(file_obj).load()


class MultimodalStressDetector:
    """
    Refactored to act strictly as an Inference Engine that obeys the FeatureRuntimeLock.
    """
    def __init__(self):
        self.facial_model = None
        self.voice_model = None
        self.phys_model = None
        
        self.facial_scaler = None
        self.voice_scaler = None
        self.phys_scaler = None
        
        self.is_trained = False
        
        self.face_extractor = FaceExtractor()
        self.voice_extractor = VoiceExtractor()
        
        # Ensures all runtime vectors match the exact training transformations
        self.feature_lock = FeatureRuntimeLock("contracts/feature_contract.yaml")

    def load_model(self, base_path='.'):
        """Loads models (Note: Phase 7 will replace this with VersionRegistry)"""
        self.load_errors = {}
        models_dir = os.path.join(base_path, 'expert_models')
        
        def _safe_load(path, name):
            if not os.path.exists(path):
                self.load_errors[name] = f"File not found: {path}"
                return None
            try:
                with open(path, 'rb') as f:
                    return safe_pickle_load(f)
            except Exception as e:
                self.load_errors[name] = str(e)
                print(f"Error unpickling {name} from {path}: {e}")
                return None

        try:
            face_path = os.path.join(models_dir, 'face_expert_lightweight.pkl')
            face_scaler_path = os.path.join(models_dir, 'face_scaler_lightweight.pkl')
            self.facial_model = _safe_load(face_path, 'facial_model')
            self.facial_scaler = _safe_load(face_scaler_path, 'facial_scaler')
            
            voice_path = os.path.join(models_dir, 'voice_expert_lightweight.pkl')
            voice_scaler_path = os.path.join(models_dir, 'voice_scaler_lightweight.pkl')
            self.voice_model = _safe_load(voice_path, 'voice_model')
            self.voice_scaler = _safe_load(voice_scaler_path, 'voice_scaler')

            phys_path = os.path.join(models_dir, 'physio_expert_lightweight.pkl')
            phys_scaler_path = os.path.join(models_dir, 'physio_scaler_lightweight.pkl')
            self.phys_model = _safe_load(phys_path, 'physio_model')
            self.phys_scaler = _safe_load(phys_scaler_path, 'physio_scaler')
            
            if self.facial_model or self.voice_model or self.phys_model:
                self.is_trained = True
                return True
            return False
        except Exception as e:
            self.load_errors['general'] = str(e)
            return False

    def detect_smile(self, image_path):
        """Legacy helper for fallback thresholding."""
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_extractor.face_cascade.detectMultiScale(gray, 1.3, 5)
            # Not loading smile cascade explicitly here to keep clean, returning 0
            return 0.0
        except: return 0.0

    def extract_facial_features(self, image_path):
        """Delegates to the dedicated FaceExtractor module."""
        return self.face_extractor.extract_features(image_path)

    def extract_voice_features(self, audio_path):
        """Delegates to the dedicated VoiceExtractor module."""
        return self.voice_extractor.extract_features(audio_path)

    def extract_physiological_features(self, eeg_data=None, gsr_data=None):
        """Legacy stub — real-time physio handled externally via PhysioExtractor."""
        return np.zeros(5)

    def predict(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0.5):
        if not self.is_trained: return {'error': 'Models not loaded'}
        
        # Phase 4-validated optimal weights: Face=0.30, Voice=0.40, Physio=0.30
        FUSION_WEIGHTS = {'facial': 0.30, 'voice': 0.40, 'physiological': 0.30}
        
        raw_probs = {}
        preds = {'facial': None, 'voice': None, 'physiological': None}
        
        # 1. Facial Expert — through FeatureRuntimeLock
        if facial_features is not None and self.facial_model:
            try:
                ff_locked = self.feature_lock.process_face_features(facial_features, self.facial_scaler)
                f_prob = self.facial_model.predict_proba(ff_locked)[0][1]
                preds['facial'] = f_prob
                raw_probs['facial'] = f_prob
            except Exception as e: print(f"Facial pred error: {e}")
            
        # 2. Voice Expert — through FeatureRuntimeLock
        if voice_features is not None and self.voice_model:
            try:
                vf_locked = self.feature_lock.process_voice_features(voice_features, self.voice_scaler)
                v_prob = self.voice_model.predict_proba(vf_locked)[0][1]
                preds['voice'] = v_prob
                raw_probs['voice'] = v_prob
            except Exception as e: print(f"Voice pred error: {e}")

        # 3. Physio Expert — now through FeatureRuntimeLock (Phase 5 integration)
        if phys_features is not None and self.phys_model:
            try:
                pf_locked = self.feature_lock.process_physio_features(phys_features, self.phys_scaler)
                p_prob = self.phys_model.predict_proba(pf_locked)[0][1]
                preds['physiological'] = p_prob
                raw_probs['physiological'] = p_prob
            except Exception as e: print(f"Physio pred error: {e}")

        if not raw_probs: return {'error': 'No valid predictions'}

        # Weighted late-fusion using Phase 4 optimal weights
        # Only include weights for active modalities, then re-normalise
        active_weights = {m: FUSION_WEIGHTS[m] for m in raw_probs if m in FUSION_WEIGHTS}
        total_w = sum(active_weights.values())
        norm_weights = {m: w / total_w for m, w in active_weights.items()}
        avg_prob = sum(raw_probs[m] * norm_weights[m] for m in raw_probs)

        threshold = 0.6 + (0.5 - sensitivity) * 0.4
        final_pred = 1 if avg_prob > threshold else 0
        stress_level = "High" if avg_prob > 0.7 else "Moderate" if avg_prob > 0.4 else "Low"

        return {
            'status': 'success',
            'predicted_class': 'Stress' if final_pred else 'No Stress',
            'stress_probability': float(avg_prob),
            'no_stress_probability': float(1 - avg_prob),
            'confidence': float(max(avg_prob, 1 - avg_prob)),
            'stress_level': stress_level,
            'percentage': float(avg_prob * 100),
            'individual_predictions': preds,
            'fusion_weights': {m: round(norm_weights.get(m, 0.0), 3) for m in preds},
        }

def fuse_predictions(probs, confs, fusion_mode='reliability'):
    """
    Phase 5: Uses Phase 4-validated optimal weights (Face=0.30, Voice=0.40, Physio=0.30).
    Only active modalities contribute; weights are re-normalised when some are absent.
    """
    OPTIMAL_WEIGHTS = {'face': 0.30, 'voice': 0.40, 'physio': 0.30}
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
        
    active_modes = [m for m in active_modes if m in OPTIMAL_WEIGHTS]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
    
    if len(active_modes) == 1:
        score = probs[active_modes[0]]
        level = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
        return {'fused_score': score, 'stress_level': level, 'weights': {active_modes[0]: 1.0}, 'modality_weights': {active_modes[0]: 1.0}}
        
    # Apply Phase 4 weights (re-normalise so active subset always sums to 1)
    raw_weights = {m: OPTIMAL_WEIGHTS[m] for m in active_modes}
    w_sum = sum(raw_weights.values())
    norm_weights = {m: raw_weights[m] / w_sum for m in active_modes}
    
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    
    rounded_weights = {m: round(w, 3) for m, w in norm_weights.items()}
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': rounded_weights,
        'modality_weights': rounded_weights
    }
