import os
import cv2
import numpy as np
import mediapipe as mp
import sys

# Check if we can use legacy solutions or modern Tasks API
USE_LEGACY_MEDIAPIPE = False
try:
    import mediapipe.solutions.face_mesh as mp_face_mesh
    USE_LEGACY_MEDIAPIPE = True
except (ImportError, AttributeError):
    USE_LEGACY_MEDIAPIPE = False

class FaceMeshWrapper:
    def __init__(self, static_mode=True):
        self.static_mode = static_mode
        self.use_tasks = not USE_LEGACY_MEDIAPIPE
        self.fm = None
        self.detector = None
        
        if self.use_tasks:
            # We assume model.py's task file logic is moved here or just expected in backend/
            # For robustness, we check the backend root.
            self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "face_landmarker.task")
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

class FaceExtractor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_mesh = FaceMeshWrapper(static_mode=False)
        
    def extract_features(self, image_path):
        """Extracts the raw 18 geometric face features. No scaling is performed here."""
        try:
            img = cv2.imread(image_path)
            if img is None: return None, None
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = img.shape
            
            results = self.face_mesh.process(rgb_img)
            
            if not results.multi_face_landmarks:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    return np.zeros(18), None
                x, y, w, h = faces[0]
                return np.zeros(18), (int(x), int(y), int(w), int(h))

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
                0.0,                                                    
                dist(55, 159) / faceH,                             
                dist(285, 386) / faceH,                            
                abs(dist(55, 159) - dist(285, 386)) / faceH,    
                dist(13, 14) / (dist(61, 291) + 1e-6),        
                dist(4, 152) / iod,                                
                (dist(61, 4) + dist(291, 4)) / (2 * faceH),   
                dist(10, 151) / faceH,                             
                faceH / iod,                                            
                0.0,                                                    
                0.0,                                                    
                0.0,                                                    
                avgEAR,                                                 
                0.9,                                                    
                dist(4, 50) / faceH,                               
            ]

            features = np.array(geom_features)
            
            x_coords = [p.x for p in landmarks]
            y_coords = [p.y for p in landmarks]
            x, y = int(min(x_coords) * w_img), int(min(y_coords) * h_img)
            w, h = int((max(x_coords) - min(x_coords)) * w_img), int((max(y_coords) - min(y_coords)) * h_img)

            return features, (x, y, w, h)
            
        except Exception as e:
            print(f"FaceExtractor Error: {e}")
            return np.zeros(18), None
