"""
Fixed model.py with correct feature dimensions matching training data
"""

import numpy as np
import cv2
import librosa
import pickle
import os
import mediapipe as mp
import sys

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

# Check if we can use legacy solutions or modern Tasks API
USE_LEGACY_MEDIAPIPE = False
try:
    # pyrefly: ignore [missing-import]
    import mediapipe.solutions.face_mesh as mp_face_mesh
    USE_LEGACY_MEDIAPIPE = True
except (ImportError, AttributeError):
    USE_LEGACY_MEDIAPIPE = False

# --- FaceMeshWrapper to support both Legacy solutions and Modern Tasks API ---
class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None
        
        if self.use_tasks:
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
            if not os.path.exists(self.model_path):
                import urllib.request
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                print("Downloading Face Landmarker model asset for Tasks API...")
                urllib.request.urlretrieve(url, self.model_path)
            
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            # pyrefly: ignore [missing-import]
            import mediapipe.solutions.face_mesh as mp_face_mesh
            self.fm = mp_face_mesh.FaceMesh(
                static_image_mode=self.static_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def process(self, rgb_image):
        if self.use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            res = self.detector.detect(mp_image)
            
            class LegacyLandmarkResult:
                def __init__(self, landmarks):
                    self.landmark = landmarks
                    
            class LegacyResult:
                def __init__(self, face_landmarks):
                    self.multi_face_landmarks = [LegacyLandmarkResult(face_landmarks[0])]
                    
            class LegacyResultEmpty:
                multi_face_landmarks = None

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                return LegacyResult(res.face_landmarks)
            else:
                return LegacyResultEmpty()
        else:
            return self.fm.process(rgb_image)

    def close(self):
        if self.fm:
            self.fm.close()
        if self.detector:
            self.detector.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MultimodalStressDetector:
    def __init__(self):
        self.facial_model = None
        self.voice_model = None
        self.phys_model = None
        
        self.facial_scaler = None
        self.voice_scaler = None
        self.phys_scaler = None
        
        self.is_trained = False
        
        # Load Haar Cascades (Legacy Backup)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        # Advanced MediaPipe Initialization
        self.face_mesh = FaceMeshWrapper(static_mode=False)
    
    def load_model(self, base_path='.'):
        """Load the 3 specialized expert models"""
        self.load_errors = {}
        models_dir = os.path.join(base_path, 'expert_models')
        
        def _safe_load(path, name):
            if not os.path.exists(path):
                self.load_errors[name] = f"File not found: {path}"
                return None
            try:
                with open(path, 'rb') as f:
                    loaded_obj = safe_pickle_load(f)
                return loaded_obj
            except Exception as e:
                err_msg = f"Pickle load error: {e}"
                self.load_errors[name] = err_msg
                print(f"Error unpickling {name} from {path}: {e}")
                return None

        try:
            # 1. Facial Expert
            face_path = os.path.join(models_dir, 'face_expert_lightweight.pkl')
            face_scaler_path = os.path.join(models_dir, 'face_scaler_lightweight.pkl')
            self.facial_model = _safe_load(face_path, 'facial_model')
            self.facial_scaler = _safe_load(face_scaler_path, 'facial_scaler')
            if self.facial_model:
                print("Loaded Facial Expert Model")
            
            # 2. Voice Expert
            voice_path = os.path.join(models_dir, 'voice_expert_lightweight.pkl')
            voice_scaler_path = os.path.join(models_dir, 'voice_scaler_lightweight.pkl')
            self.voice_model = _safe_load(voice_path, 'voice_model')
            self.voice_scaler = _safe_load(voice_scaler_path, 'voice_scaler')
            if self.voice_model:
                print("Loaded Voice Expert Model")

            # 3. Physio Expert
            phys_path = os.path.join(models_dir, 'physio_expert.pkl')
            phys_scaler_path = os.path.join(models_dir, 'physio_scaler.pkl')
            self.phys_model = _safe_load(phys_path, 'physio_model')
            self.phys_scaler = _safe_load(phys_scaler_path, 'physio_scaler')
            if self.phys_model:
                print("Loaded Physio Expert Model")
            
            # Allow the detector to function if at least one model is successfully loaded
            if self.facial_model or self.voice_model or self.phys_model:
                self.is_trained = True
                return True
            else:
                self.is_trained = False
                return False
        except Exception as e:
            self.load_errors['general'] = str(e)
            print(f"Error loading models: {e}")
            return False

    def detect_smile(self, image_path):
        """Heuristic to detect if a person is smiling."""
        try:
            img = cv2.imread(image_path)
            if img is None: return 0.0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0: return 1.0
            return 0.0
        except: return 0.0

    def extract_facial_features(self, image_path):
        """Advanced MediaPipe-based Feature Extraction (18 Features for Expert Model)"""
        try:
            img = cv2.imread(image_path)
            if img is None: return None, None
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = img.shape
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                # Fallback to Haar for bounding box if MediaPipe fails to find mesh
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return np.zeros(18), None
                x, y, w, h = faces[0]
                return np.zeros(18), (int(x), int(y), int(w), int(h))

            # 1. High Level Geometric Features (18 features)
            landmarks = results.multi_face_landmarks[0].landmark
            pts = np.array([[l.x * w_img, l.y * h_img] for l in landmarks])

            def dist(p1_idx, p2_idx):
                return np.sqrt((pts[p1_idx][0] - pts[p2_idx][0])**2 + (pts[p1_idx][1] - pts[p2_idx][1])**2)

            faceH = dist(10, 152) + 1e-6
            faceW = dist(234, 454) + 1e-6
            iod   = dist(33, 263) + 1e-6

            earL = (dist(159, 145) + dist(158, 153)) / (2 * dist(33, 133) + 1e-6)
            earR = (dist(386, 374) + dist(385, 380)) / (2 * dist(362, 263) + 1e-6)
            avgEAR = (earL + earR) / 2

            geom_features = [
                earL,
                earR,
                avgEAR,
                0.0,                                                    # blink_velocity
                dist(55, 159) / faceH,                             # brow_descent_left
                dist(285, 386) / faceH,                            # brow_descent_right
                abs(dist(55, 159) - dist(285, 386)) / faceH,    # brow_asymmetry
                dist(13, 14) / (dist(61, 291) + 1e-6),        # lip_compression
                dist(4, 152) / iod,                                # jaw_displacement
                (dist(61, 4) + dist(291, 4)) / (2 * faceH),   # mouth_corner_pull
                dist(10, 151) / faceH,                             # forehead_tension
                faceH / iod,                                            # face_height_norm
                0.0,                                                    # head_tilt
                0.0,                                                    # temporal_x_var
                0.0,                                                    # temporal_y_var
                avgEAR,                                                 # eye_openness_ratio
                0.9,                                                    # landmark_confidence
                dist(4, 50) / faceH,                               # nose_wrinkle
            ]

            features = np.array(geom_features)
            
            # Find bounding box from landmarks
            x_coords = [p.x for p in landmarks]
            y_coords = [p.y for p in landmarks]
            x, y = int(min(x_coords) * w_img), int(min(y_coords) * h_img)
            w, h = int((max(x_coords) - min(x_coords)) * w_img), int((max(y_coords) - min(y_coords)) * h_img)

            return features, (x, y, w, h)
            
        except Exception as e:
            print(f"Error extracting facial features: {e}")
            return np.zeros(18), None

    def extract_voice_features(self, audio_path):
        """Extract 12 audio features from a WAV file using voice_worker logic."""
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            from voice_worker import extract_voice_stress_indicators
            res = extract_voice_stress_indicators(audio_bytes)
            if res is not None:
                return res['features']
            return None
        except Exception as e:
            print(f"Voice feature extraction error: {e}")
            return None

    def extract_physiological_features(self, eeg_data=None, gsr_data=None):
        # Delegate to the module-level function for 51 features
        return extract_physiological_features(eeg_data, gsr_data)

    def predict(self, facial_features=None, voice_features=None, phys_features=None, temp_image_path=None, sensitivity=0.5):
        if not self.is_trained: return {'error': 'Models not loaded'}
        
        probs = []
        preds = {'facial': None, 'voice': None, 'physiological': None}
        
        # 1. Facial Expert
        if facial_features is not None and self.facial_model:
            try:
                ff = np.array(facial_features).reshape(1, -1)
                if self.facial_scaler:
                    ff = self.facial_scaler.transform(ff)
                f_prob = self.facial_model.predict_proba(ff)[0][1]
                preds['facial'] = f_prob
                
                if temp_image_path:
                    smile = self.detect_smile(temp_image_path)
                    if smile > 0.5:
                        f_prob = max(0.0, f_prob - 0.4) 
                
                probs.append(f_prob)
            except Exception as e: print(f"Facial pred error: {e}")
            
        # 2. Voice Expert
        if voice_features is not None and self.voice_model:
            try:
                vf = np.array(voice_features).reshape(1, -1)
                vf_scaled = self.voice_scaler.transform(vf)
                v_prob = self.voice_model.predict_proba(vf_scaled)[0][1]
                preds['voice'] = v_prob
                probs.append(v_prob)
            except Exception as e: print(f"Voice pred error: {e}")

        # 3. Physio Expert
        if phys_features is not None and self.phys_model:
            try:
                pf = np.array(phys_features).reshape(1, -1)
                pf_scaled = self.phys_scaler.transform(pf)
                p_prob = self.phys_model.predict_proba(pf_scaled)[0][1]
                preds['physiological'] = p_prob
                probs.append(p_prob)
            except Exception as e: print(f"Physio pred error: {e}")

        if not probs: return {'error': 'No valid predictions'}

        # Simple Average Fusion
        avg_prob = np.mean(probs)
        
        # Sensitivity Thresholding
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
            'individual_predictions': preds
        }


# --- Module-Level Physiological Feature Extractors and Fusion ---

def extract_eeg_features(eeg_data, fs=256):
    import numpy as np
    from scipy import signal
    
    if eeg_data is None or len(eeg_data) == 0:
        return np.zeros(42)
        
    eeg_arr = np.array(eeg_data)
    
    # Compute PSD using Welch's method
    nperseg = min(len(eeg_arr), 256)
    if nperseg < 8:
        f, psd = np.array([0]), np.array([0])
    else:
        f, psd = signal.welch(eeg_arr, fs, nperseg=nperseg)
        
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    feats = np.zeros(42)
    band_means = {}
    for i, (band_name, (low, high)) in enumerate(bands.items()):
        idx = np.where((f >= low) & (f <= high))[0]
        if len(idx) > 0:
            bp_mean = float(np.mean(psd[idx]))
            bp_std = float(np.std(psd[idx]))
        else:
            bp_mean = 0.0
            bp_std = 0.0
            
        band_means[band_name] = bp_mean
        feats[2 * i] = bp_mean
        feats[2 * i + 1] = bp_std
        
    # Stress index: feats[11] = (beta + gamma) / (alpha + theta)
    denom = band_means['alpha'] + band_means['theta']
    if denom > 1e-10:
        feats[11] = (band_means['beta'] + band_means['gamma']) / denom
    else:
        feats[11] = 0.0
        
    return feats

def extract_gsr_features(gsr_data, fs=4):
    import numpy as np
    if gsr_data is None or len(gsr_data) == 0:
        return np.zeros(9)
        
    gsr_arr = np.array(gsr_data)
    
    # Smooth gsr to avoid noise peaks
    if len(gsr_arr) > 5:
        gsr_smooth = np.convolve(gsr_arr, np.ones(5)/5, mode='valid')
    else:
        gsr_smooth = gsr_arr
        
    diffs = np.diff(gsr_smooth)
    peaks = 0
    in_peak = False
    for val in diffs:
        if val > 0.05:
            if not in_peak:
                peaks += 1
                in_peak = True
        else:
            in_peak = False
            
    feats = np.zeros(9)
    feats[0] = float(np.mean(gsr_arr))
    feats[1] = float(np.std(gsr_arr))
    feats[2] = float(np.max(gsr_arr) - np.min(gsr_arr))
    feats[3] = float(peaks)
    
    return feats

def extract_physiological_features(eeg_data=None, gsr_data=None):
    import numpy as np
    # Standard 42 EEG + 9 GSR features (total 51)
    eeg_feats = extract_eeg_features(eeg_data)
    gsr_feats = extract_gsr_features(gsr_data)
    return np.concatenate([eeg_feats, gsr_feats])

def fuse_predictions(probs, confs, fusion_mode='reliability'):
    active_modes = list(probs.keys())
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
        
    base_weights = {'face': 0.371, 'voice': 0.474, 'physio': 0.338}
    
    active_modes = [m for m in active_modes if m in base_weights]
    if not active_modes:
        return {'fused_score': 0.0, 'stress_level': 'Low', 'weights': {}, 'modality_weights': {}}
        
    raw_weights = {}
    for m in active_modes:
        raw_weights[m] = base_weights[m] * confs.get(m, 1.0)
        
    w_sum = sum(raw_weights.values())
    if w_sum > 0:
        norm_weights = {m: raw_weights[m] / w_sum for m in active_modes}
    else:
        norm_weights = {m: 1.0 / len(active_modes) for m in active_modes}
        
    rounded_weights = {m: round(w, 3) for m, w in norm_weights.items()}
    fused_score = sum(probs[m] * norm_weights[m] for m in active_modes)
    level = "High" if fused_score > 0.7 else "Moderate" if fused_score > 0.4 else "Low"
    
    return {
        'fused_score': fused_score,
        'stress_level': level,
        'weights': rounded_weights,
        'modality_weights': rounded_weights
    }

