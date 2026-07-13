import numpy as np
import os

class VoiceExtractor:
    """Wrapper that delegates to the specialized voice_worker for audio feature extraction."""
    def extract_features(self, audio_path):
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
                
            # Local import to avoid circular dependencies
            import sys
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if backend_dir not in sys.path:
                sys.path.append(backend_dir)
                
            from voice_worker import extract_voice_stress_indicators
            res = extract_voice_stress_indicators(audio_bytes)
            if res is not None:
                return np.array(res['features'])
            return None
        except Exception as e:
            print(f"VoiceExtractor Error: {e}")
            return None
