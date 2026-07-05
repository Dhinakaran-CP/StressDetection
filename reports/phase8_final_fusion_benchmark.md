# Phase 8: Final Audited Multimodal Fusion Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Modality Encoders**: PyTorch 1D-CNN+GRU Encoders (Face, Physio) trained with Time Masking augmentation.
- **Fusion Engine**: Dynamic Router MLP (Face + Physio probabilities gate weights).
- **Sequence Length**: 5

## Final Benchmark Results
| Modality/Strategy | Mean Accuracy | Std Dev |
|-------------------|---------------|---------|
| Face-Only Encoder | 0.5912 | 0.0747 |
| Physio-Only Encoder | 0.5485 | 0.0852 |
| **Dynamic Pairwise Fusion** | **0.5875** | **0.0847** |
