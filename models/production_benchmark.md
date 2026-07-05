# Production Models Multi-Strategy Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Feature Contract**: Standard normalized calibration inputs
- **Sequence Length**: 5

## Strategy 4 (Standard CNN-GRU) Benchmarks
- **Face-Only**: 0.6614 ($\pm$ 0.0338)
- **Voice-Only**: 0.6243 ($\pm$ 0.0459)
- **Physio-Only**: 0.6556 ($\pm$ 0.0297)
- **3-Way Fusion**: **0.6724** ($\pm$ 0.0233)

## Strategy 5 (Adversarial CNN-GRU) Benchmarks (PRIMARY)
- **Face-Only**: 0.6706 ($\pm$ 0.0301)
- **Voice-Only**: 0.6186 ($\pm$ 0.0281)
- **Physio-Only**: 0.6424 ($\pm$ 0.0241)
- **3-Way Fusion (Adversarial)**: **0.6736** ($\pm$ 0.0384)
