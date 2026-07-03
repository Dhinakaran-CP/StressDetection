import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

# Create dummy wav bytes
sr = 16000
y = np.zeros(16000, dtype=np.float32)
fd, temp_wav = tempfile.mkstemp(suffix='.wav')
os.close(fd)
sf.write(temp_wav, y, sr)

with open(temp_wav, 'rb') as f:
    audio_bytes = f.read()

# Fallback mechanism in voice_worker.py
try:
    fd, temp_path = tempfile.mkstemp(suffix='.bin')
    os.close(fd)
    with open(temp_path, 'wb') as f:
        f.write(audio_bytes)
    
    y, sr = librosa.load(temp_path, sr=16000, mono=True, duration=3.0)
    print("Fallback successful!")
except Exception as e:
    print("Fallback failed:", e)
finally:
    os.remove(temp_wav)
    if os.path.exists(temp_path):
        os.remove(temp_path)
