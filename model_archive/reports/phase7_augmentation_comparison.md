# Phase 7: Augmentation Comparison Report

## Protocol
- **Validation**: Leave-One-Subject-Out (Strict 5-Fold GroupKFold, Subset 15 Subjects)
- **Model**: Gated Fusion Network (1D-CNN+GRU encoders for Face & Physio + dynamic router)
- **Epochs**: 8

## Results
| Augmentation Strategy | Mean Accuracy | Std Dev | Performance Delta |
|-----------------------|---------------|---------|-------------------|
| None | 0.6260 | 0.0704 | +0.0000 |
| Jitter | 0.6124 | 0.0670 | -0.0136 |
| Scaling | 0.6034 | 0.0879 | -0.0226 |
| Time_mask | 0.6316 | 0.0553 | +0.0056 |
| Modality_dropout | 0.6080 | 0.0844 | -0.0180 |
| Combined | 0.6191 | 0.0696 | -0.0069 |

## Conclusion
Choose the simplest augmentation method that consistently improves validation accuracy or reduces standard deviation. If no augmentation shows benefits on validation set (meaning delta is negative or zero), we reject it to avoid unnecessary runtime/training overhead.
