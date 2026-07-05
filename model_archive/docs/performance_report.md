
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 31.35 ms
- **Min Latency**: 19.42 ms
- **Max Latency**: 174.28 ms
- **FPS Capacity**: 31.9 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 549.99 ms
- **Min Latency**: 182.25 ms
- **Max Latency**: 4961.32 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: NO)
    