import numpy as np
import soundfile as sf
import io
import pickle
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

voice_expert = pickle.load(open('expert_models/voice_expert_lightweight.pkl','rb'))
voice_scaler  = pickle.load(open('expert_models/voice_scaler_lightweight.pkl','rb'))

from voice_worker import extract_voice_stress_indicators

# Test 1: Complete silence
sr = 16000
silence = np.zeros(sr * 2, dtype=np.float32)
buf = io.BytesIO()
sf.write(buf, silence, sr, format='WAV')
result = extract_voice_stress_indicators(buf.getvalue())
if result:
    scaled = voice_scaler.transform(result['features'].reshape(1,-1))
    score = float(voice_expert.predict_proba(scaled)[0][1])
    print(f"Pure silence score: {score:.4f}  (should be None from silence gate)")
else:
    print("Pure silence: correctly returned None")

# Test 2: Calm 150Hz tone
t = np.linspace(0, 2, sr*2)
calm_tone = (0.3 * np.sin(2*np.pi*150*t)).astype(np.float32)
buf2 = io.BytesIO()
sf.write(buf2, calm_tone, sr, format='WAV')
result2 = extract_voice_stress_indicators(buf2.getvalue())
if result2:
    print(f"\nCalm 150Hz tone indicators:")
    for k,v in result2['indicators'].items():
        print(f"  {k}: {v:.4f}")
    scaled2 = voice_scaler.transform(result2['features'].reshape(1,-1))
    score2 = float(voice_expert.predict_proba(scaled2)[0][1])
    print(f"\nCalm 150Hz tone score: {score2:.4f}  (should be 0.20-0.45)")
    if score2 > 0.80:
        print("PROBLEM: Voice model biased toward stress prediction")
        print("Cause: Training data imbalance OR scaler mean/std mismatch")
        print("Check: voice_scaler.mean_ - should match StressID calm voice stats")
        print(f"voice_scaler.mean_: {voice_scaler.mean_}")
        print(f"voice_scaler.scale_: {voice_scaler.scale_}")
