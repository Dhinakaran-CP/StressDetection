# Phase 8: Final Audited Multimodal Fusion Benchmark

## Protocol
- **Validation**: Strict Leave-One-Subject-Out (5-Fold GroupKFold) on Full 65 Subjects
- **Modality Encoders**: PyTorch 1D-CNN+GRU Encoders trained with Time Masking augmentation.
- **Fusion Engine**: Flex-Modality Dynamic Router MLP (supports any subset of Face, Voice, Physio inputs).
- **Sequence Length**: 5

## Final Benchmark Results (Cross-Subject Validation Accuracies)

### Unimodal Encoders
- **Face-Only**: 0.5510 ($\pm$ 0.0458)
- **Voice-Only**: 0.6146 ($\pm$ 0.0314)
- **Physio-Only**: 0.5895 ($\pm$ 0.0448)

### Pairwise Combinations
- **Face + Physio**: 0.5789 ($\pm$ 0.0370)
- **Face + Voice**: 0.5557 ($\pm$ 0.0386)
- **Voice + Physio**: 0.5827 ($\pm$ 0.0253)

### Full 3-Way Fusion
- **Face + Voice + Physio (All Sensors Present)**: **0.5826** ($\pm$ 0.0303)
