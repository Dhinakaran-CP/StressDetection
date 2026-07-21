import cv2
import mediapipe as mp
import librosa
import numpy as np

print("Imports OK")
print(f"MediaPipe: {dir(mp)}")
print(f"Has solutions: {hasattr(mp, 'solutions')}")

# Test voice extraction
y = np.zeros(1000)
sr = 22050
try:
    # Use hop_length that might fail
    hop_length = int(sr / 3)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    print("Voice test OK")
except Exception as e:
    print(f"Voice test failed: {e}")
