
import time
import numpy as np
import cv2
import base64
import json
import matplotlib.pyplot as plt
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.realtime_core import StressStreamProcessor

def generate_dummy_image_base64(width=320, height=240):
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def generate_dummy_audio_chunk(sr=44100, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio.astype(np.float32)

def test_performance():
    print("Initializing Real-time Processor...")
    processor = StressStreamProcessor()
    
    n_iterations = 50
    video_latencies = []
    audio_latencies = []
    
    print(f"\nRunning {n_iterations} iterations for Video pipeline...")
    dummy_img = generate_dummy_image_base64()
    
    for i in range(n_iterations):
        start = time.time()
        result = processor.process_video_frame(dummy_img)
        end = time.time()
        latency = (end - start) * 1000 # ms
        video_latencies.append(latency)
        if i % 10 == 0:
            print(f"  Frame {i}: {latency:.2f} ms")
            
    print(f"\nRunning {n_iterations} iterations for Audio pipeline...")
    dummy_audio = generate_dummy_audio_chunk(duration=1.0) # 1 sec chunk
    
    # Initialize session
    session_id = 'test_session'
    processor.initialize_session(session_id)
    
    for i in range(n_iterations):
        # We need to feed enough audio to trigger prediction (MIN_AUDIO_FOR_PREDICTION = 1.0)
        # So each chunk is processed.
        start = time.time()
        result = processor.process_audio_chunk(session_id, dummy_audio)
        end = time.time()
        latency = (end - start) * 1000 # ms
        audio_latencies.append(latency)
        if i % 10 == 0:
            print(f"  Chunk {i}: {latency:.2f} ms")

    # Generate Report
    avg_video = np.mean(video_latencies)
    avg_audio = np.mean(audio_latencies)
    
    report = f"""
# Real-Time Performance Report

## Summary
- **Iterations**: {n_iterations}
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: {avg_video:.2f} ms
- **Min Latency**: {np.min(video_latencies):.2f} ms
- **Max Latency**: {np.max(video_latencies):.2f} ms
- **FPS Capacity**: {1000/avg_video:.1f} FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: {avg_audio:.2f} ms
- **Min Latency**: {np.min(audio_latencies):.2f} ms
- **Max Latency**: {np.max(audio_latencies):.2f} ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: {'YES' if avg_video < 200 else 'NO'})
- **Target Audio Latency**: < 500ms (Achieved: {'YES' if avg_audio < 500 else 'NO'})
    """
    
    print(report)
    
    with open("performance_report.md", "w") as f:
        f.write(report)
        
    # Generate Charts
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(video_latencies)
    plt.title('Video Processing Latency')
    plt.xlabel('Frame')
    plt.ylabel('Time (ms)')
    plt.axhline(y=avg_video, color='r', linestyle='--', label=f'Avg: {avg_video:.1f}ms')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(audio_latencies)
    plt.title('Audio Processing Latency')
    plt.xlabel('Chunk')
    plt.ylabel('Time (ms)')
    plt.axhline(y=avg_audio, color='r', linestyle='--', label=f'Avg: {avg_audio:.1f}ms')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_charts.png')
    print("Charts saved to performance_charts.png")

if __name__ == "__main__":
    test_performance()
