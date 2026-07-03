import requests
import time
import numpy as np
import io
import soundfile as sf

def test_api():
    print("=== Testing Multimodal HTTP API Endpoints ===")
    
    # 1. Test Face Stream Endpoint
    print("\n1. Testing /api/stream/face...")
    face_payload = {
        "indicators": {
            "left_ear": 0.32,
            "right_ear": 0.31,
            "avg_ear": 0.315,
            "blink_velocity": 1.2,
            "brow_descent_left": 0.12,
            "brow_descent_right": 0.11,
            "brow_asymmetry": 0.01,
            "lip_compression": 0.22,
            "jaw_tension": 0.68,
            "mouth_corner_pull": 0.28,
            "forehead_tension": 0.12,
            "face_height_norm": 1.48,
            "head_tilt": 2.1,
            "temporal_x_var": 0.002,
            "temporal_y_var": 0.003,
            "eye_openness_ratio": 0.315,
            "landmark_confidence": 0.95,
            "nose_wrinkle": 0.11
        },
        "timestamp": int(time.time() * 1000)
    }
    
    try:
        r_face = requests.post("http://127.0.0.1:5000/api/stream/face", json=face_payload)
        print("Status Code:", r_face.status_code)
        print("Response JSON:", r_face.json())
        assert r_face.status_code == 200
        assert "score" in r_face.json()
    except Exception as e:
        print("Face API test failed:", e)
        return

    # 2. Test Voice Stream Endpoint
    print("\n2. Testing /api/stream/voice...")
    # Generate 2 seconds of fake voiced sound (e.g. 200Hz sine wave)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    # Adding some harmonics to ensure F0 is detected
    y = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    
    # Save to a virtual WAV file buffer
    wav_io = io.BytesIO()
    sf.write(wav_io, y, sr, format='WAV', subtype='PCM_16')
    wav_bytes = wav_io.getvalue()
    
    try:
        r_voice = requests.post(
            "http://127.0.0.1:5000/api/stream/voice",
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"}
        )
        print("Status Code:", r_voice.status_code)
        print("Response JSON:", r_voice.json())
        assert r_voice.status_code == 200
        assert "score" in r_voice.json()
    except Exception as e:
        print("Voice API test failed:", e)
        return

    # 3. Test Fused SSE Stream Endpoint
    print("\n3. Testing /api/stream/fused (SSE)...")
    try:
        # Read the first few lines of the SSE stream
        r_fused = requests.get("http://127.0.0.1:5000/api/stream/fused", stream=True, timeout=5)
        print("Status Code:", r_fused.status_code)
        assert r_fused.status_code == 200
        
        line_count = 0
        for line in r_fused.iter_lines(chunk_size=1):
            if line:
                line_str = line.decode('utf-8')
                print("SSE Line:", line_str)
                if line_str.startswith("data:"):
                    line_count += 1
                    if line_count >= 2:
                        break
    except Exception as e:
        print("Fused SSE API test failed:", e)
        return

    print("\n=== All HTTP API Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api()
