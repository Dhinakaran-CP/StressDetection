
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 176.43 ms
- **Min Latency**: 85.17 ms
- **Max Latency**: 568.84 ms
- **FPS Capacity**: 5.7 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 367.21 ms
- **Min Latency**: 120.95 ms
- **Max Latency**: 10082.40 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: YES)
    