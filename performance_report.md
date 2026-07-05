
# Real-Time Performance Report

## Summary
- **Iterations**: 50
- **Video Resolution**: 320x240
- **Audio Chunk**: 1.0 second (44.1kHz)

## Results
### Video Pipeline (Image -> Features -> Predict)
- **Average Latency**: 73.54 ms
- **Min Latency**: 35.17 ms
- **Max Latency**: 657.79 ms
- **FPS Capacity**: 13.6 FPS

### Audio Pipeline (Waveform -> Features -> Predict)
- **Average Latency**: 911.37 ms
- **Min Latency**: 240.12 ms
- **Max Latency**: 10936.88 ms

## Benchmark
- **Target Video Latency**: < 200ms (Achieved: YES)
- **Target Audio Latency**: < 500ms (Achieved: NO)
    