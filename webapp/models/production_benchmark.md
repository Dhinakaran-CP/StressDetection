# Production Models Multi-Strategy Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Feature Contract**: Standard normalized calibration inputs
- **Sequence Length**: 5

## Strategy 4 (Standard CNN-GRU) Benchmarks
- **Face-Only**: 0.6338 ($\pm$ 0.0170)
- **Voice-Only**: 0.6772 ($\pm$ 0.0338)
- **Physio-Only**: 0.6430 ($\pm$ 0.0261)
- **3-Way Fusion**: **0.6944** ($\pm$ 0.0163)

## Strategy 5 (Adversarial CNN-GRU) Benchmarks (PRIMARY)
- **Face-Only**: 0.6603 ($\pm$ 0.0136)
- **Voice-Only**: 0.6816 ($\pm$ 0.0481)
- **Physio-Only**: 0.6603 ($\pm$ 0.0151)
- **3-Way Fusion (Adversarial)**: **0.7051** ($\pm$ 0.0216)
