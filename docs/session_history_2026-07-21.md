# Session History — 2026-07-21

## Summary
Designed and implemented **ConvMoE-MF** (Convolutional Mixture of Experts for Multimodal Fusion), a ~10K-parameter replacement for the 500K+ param SSVB-CASA-AIS. Wrote architecture design doc, model implementation, and session history.

## Key Decisions

### Critique of existing SSVB-CASA-AIS (5 flaws identified)
1. **Overparameterization**: 500K+ params for 290 samples/fold → guaranteed overfitting
2. **Redundant cross-attention**: 6 directional attention blocks when 1 shared block suffices
3. **Dual identity suppression**: Quality gates + GRL cause gradient conflicts
4. **Ambiguous quality masks**: Undefined data flow, no ablation study
5. **Miscalibrated expert allocation**: Physio (4 features) gets same expert count as Face (33 features)

### ConvMoE-MF architecture (addressing all 5 flaws)
- **40× smaller**: 8,584 params vs 500K+
- **CNN base**: Conv1D encoders with GAP per modality (translation equivariance for time-series)
- **MoE fusion**: 4 experts + softmax router (no cross-attention, 6→0 heads)
- **GRL only**: Single identity suppression mechanism (proven effective at λ=0.02)
- **Proportional capacity**: Face gets 2 conv layers (33→16→8), Voice gets 2 (23→16→8), Physio gets 1 (13→8)

### Rejected alternatives
- **ViT/wav2vec/HuBERT**: Incompatible with hand-crafted feature vectors (69-dim, not raw pixels/audio). Would require 86M+ params fine-tuned on 290 samples.
- **1D-CNN + Transformer for physio**: Plausible but unnecessary at 3fps with 5-frame windows — Conv1D k=5 captures full temporal context.

## Sessions Log
- Aired critique of SSVB-CASA-AIS vs CNN comparison table
- Discussed hybrid CNN+MoE approach
- ConvMoE-MF design: 3 Conv1D encoders → MoE fusion (4 experts) → 3 heads
- Implemented `webapp/backend/runtime/conv_moe_mf.py` (verified: 8,584 params, all shapes correct)
- Written architecture doc at `docs/conv_moe_mf_architecture.md`

## Verified Output
```
ConvMoE-MF params: 8,584
Stress logits: torch.Size([4, 2])
Confidence:    torch.Size([4])
Subject logits:torch.Size([4, 65])
Gate weights:  torch.Size([4, 4])
Fused emb:     torch.Size([4, 8])
```
