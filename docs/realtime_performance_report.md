
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 30.92 ms
- **Min Latency**: 15.96 ms
- **Max Latency**: 645.31 ms
- **FPS Capacity**: 32.3 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 338.53 ms
- **Min Latency**: 124.60 ms
- **Max Latency**: 4876.32 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: YES)
    