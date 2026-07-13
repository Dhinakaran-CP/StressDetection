import numpy as np
import os
import tempfile
import soundfile as sf
import base64
import cv2
from backend.model import MultimodalStressDetector
from backend.runtime.session_state import SessionState

class StressStreamProcessor:
    def __init__(self, runtime_engine=None):
        # Fallback to model instance if runtime_engine isn't passed (for legacy tests)
        self.runtime_engine = runtime_engine
        if not self.runtime_engine:
            self.model = MultimodalStressDetector()
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            self.model.load_model(BASE_DIR)
        else:
            self.model = MultimodalStressDetector()  # just for extractors

        self.sessions = {}
        self.MAX_AUDIO_BUFFER_SECONDS = 3.0
        self.MIN_AUDIO_FOR_PREDICTION = 0.5 
        
    def initialize_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                max_audio_seconds=self.MAX_AUDIO_BUFFER_SECONDS,
                min_audio_seconds=self.MIN_AUDIO_FOR_PREDICTION,
                sample_rate=44100
            )
        
    def remove_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def process_video_frame(self, image_data_base64, sensitivity=0.5):
        try:
            if ',' in image_data_base64:
                image_data = image_data_base64.split(',')[1]
            else:
                image_data = image_data_base64
                
            image_bytes = base64.b64decode(image_data)
            
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            with open(temp_path, 'wb') as f:
                f.write(image_bytes)
            
            features, face_coords = self.model.extract_facial_features(temp_path)
            
            if features is None:
                 try: os.remove(temp_path) 
                 except: pass
                 return {'error': 'No face detected'}

            # Predict
            if self.runtime_engine:
                result = self.runtime_engine.predict_face(features, sensitivity=sensitivity)
                # Map to old format for frontend compatibility if needed
                if 'status' not in result and 'error' not in result:
                    result['status'] = 'success'
            else:
                result = self.model.predict(facial_features=features, temp_image_path=temp_path, sensitivity=sensitivity)
            
            if face_coords:
                result['face_box'] = face_coords
            
            try: os.remove(temp_path)
            except: pass
                
            return result
            
        except Exception as e:
            print(f"Video processing error: {e}")
            return {'error': str(e)}

    def process_audio_chunk(self, session_id, audio_blob, sample_rate=44100, sensitivity=0.5):
        if session_id not in self.sessions:
            self.initialize_session(session_id)
            
        session = self.sessions[session_id]
        
        try:
            buffered_audio = session.buffer_audio(audio_blob, sample_rate)
            if buffered_audio is None:
                return {'status': 'buffering', 'message': 'Collecting audio...'}
            
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            sf.write(temp_path, buffered_audio, sample_rate)
            features = self.model.extract_voice_features(temp_path)
            
            if features is None:
                try: os.remove(temp_path)
                except: pass
                return {'error': 'Silent or invalid audio chunk'}
                
            if self.runtime_engine:
                result = self.runtime_engine.predict_voice(features, sensitivity=sensitivity)
                if 'status' not in result and 'error' not in result:
                    result['status'] = 'success'
            else:
                result = self.model.predict(voice_features=features, sensitivity=sensitivity)
            
            try: os.remove(temp_path)
            except: pass
                
            return result
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return {'error': str(e)}
