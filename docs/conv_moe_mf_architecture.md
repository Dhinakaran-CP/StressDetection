# ConvMoE-MF: Convolutional Mixture of Experts for Multimodal Fusion

## Problem

Current SSVB-CASA-AIS: **~500K+ params** for **290 samples/fold** (5-fold LOSO).  
Result: Overparameterization → overfitting → poor calibration (ECE 52.57% on VBC-CASA-IS).

## Solution: ConvMoE-MF

**Replace 8 heavy SequenceExperts (Conv1D + SelfAttn + GRU each) with lightweight per-modality Conv1D backbones, then feed into a compact MoE fusion with confidence head and GRL.**

### Parameter Count Comparison

| Component | Current SSVB-CASA-AIS | ConvMoE-MF |
|-----------|----------------------|------------|
| Per-modality encoders | 8 × (Conv1D + SelfAttn + GRU) = ~480K | 3 × (2-layer Conv1D + GAP) = ~7K |
| Intra-modality gates | 3 × (Linear→ReLU→Linear) = ~2.5K | Removed |
| Cross-attention | 6 × (Q/K/V + out_proj) = ~12K | Removed (single MoE router instead) |
| MoE fusion | 1 × GlobalMoERouter = ~1.5K | 4 experts × 2-layer MLP + router = ~2K |
| Output heads | 3 × Linear = ~1.5K + 65 | 3 × Linear = ~1K + 65 |
| **Total** | **~500K+** | **~12K + 65** ≈ **12K** |

**12K params vs 500K+ → 40× reduction.**

### Why This Works Better at N=290

1. **CNN inductive bias**: Translation equivariance is correct for time-series stress signals (HR spike at any time → same pattern). No need for self-attention's global receptive field on 5-frame windows.

2. **GAP (Global Average Pooling)**: Forces the Conv1D backbone to learn per-frame features, then collapses to modality embedding. No GRU = no vanishing gradient on short sequences, fewer params.

3. **MoE instead of cross-attention**: Cross-attention (Q/K/V per pair) learns brittle pairwise relationships that don't generalize at 290 samples. MoE routers learn modality importance weights — a simpler task that converges faster.

4. **GRL only**: Your own λ_adv=0.02 sweep already proved adversarial suppression works (18.99% → 7.43% leakage gap). Quality masks + GRL = gradient conflict for no gain.

---

## Architecture Details

### Stage 1: Per-Modality Conv1D Encoders

```
Face  (34, T):  Conv1D(34→16, k=5) → BN → ReLU → Conv1D(16→8, k=3) → GAP → [8]
Voice (24, T):  Conv1D(24→16, k=5) → BN → ReLU → Conv1D(16→8, k=3) → GAP → [8]
Physio(14, T):  Conv1D(14→8,  k=3)                                  → GAP → [8]

Concatenate: [24] ← [face_8, voice_8, physio_8]
```

**Design rationale**:
- Face (34-dim, richest modality) gets 2 conv layers with sufficient capacity
- Voice (24-dim) same as face: matching capacity ensures one modality doesn't dominate
- Physio (14-dim) single conv layer: only 4 meaningful features (HR, HRVx2, resp), proved to be lower information density in your audit
- Kernel size 5 for face/voice (capture ~1.5s context at 3fps), 3 for physio (faster physiological response)
- GAP instead of GRU: 5-frame windows have limited temporal structure; GAP enforces feature-level invariance

**Per-modality forward**:
```python
class Conv1DEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 8, num_layers: int = 2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else output_dim
            k = 5 if input_dim > 20 else 3  # Larger kernel for richer modalities
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        # GAP is applied in forward, not as a layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        x = x.permute(0, 2, 1)      # [B, input_dim, T]
        x = self.net(x)              # [B, output_dim, T]
        x = x.mean(dim=2)           # [B, output_dim]  ← GAP
        return x
```

### Stage 2: Compact MoE Fusion

```
MoE Components:
  4 experts:  MLP(24→16→8)  × 4  (each sees the full [face, voice, physio] embedding)
  1 router:   MLP(24→4, softmax)

Fused = Σ_i router_weight_i × expert_i(concat_embedding)

Output: [8] ← fused embedding
```

**Design rationale**:
- 4 experts (not 8) for 290 samples: 4 experts × 16 hidden × 24 inputs = ~2K shared params  
- Router sees all 3 modalities' embeddings → learns per-sample modality weighting
- No cross-attention: MoE implicitly models modality interactions through expert specialization

```python
class MoEFusion(nn.Module):
    def __init__(self, input_dim: int = 24, num_experts: int = 4, hidden_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 24]
        weights = self.router(x)                     # [B, num_experts]
        outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, num_experts, 8]
        fused = (weights.unsqueeze(-1) * outputs).sum(dim=1)        # [B, 8]
        return fused, weights
```

### Stage 3: Output Heads

```
Stress head:      Linear(8→2)              → logits (cross-entropy)
Confidence head:  Linear(8→1) + Sigmoid    → confidence score (0-1)
Subject head:     GRL(λ=0.02) → Linear(8→N) → logits (CE for identity suppression)
```

**Design rationale**:
- Confidence head kept and unchanged from SSVB-CASA-AIS — your data already showed it correlates with per-fold accuracy (Fold 1: 75.94% / 59.19% confidence vs Fold 2: 55.78% / 52.89%)
- GRL kept as the sole identity suppression — proven effective in your λ_adv sweep
- No quality masks: removed entirely (gradient conflict risk with GRL)

### Total Model Class

```python
class ConvMoE_MF(nn.Module):
    """
    ConvMoE-MF: Convolutional Mixture of Experts for Multimodal Fusion.
    
    Architecture:
      Stage 1: Per-modality Conv1D encoders with GAP (Face→8, Voice→8, Physio→8)
      Stage 2: MoE fusion (4 experts × 24→8, router 24→4)
      Stage 3: Stress, Confidence, Adversarial Subject heads
    
    Total params: ~12K
    """
    def __init__(self, num_subjects=65, hidden_dim=16, embed_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Stage 1: Per-modality Conv1D encoders
        self.enc_face  = Conv1DEncoder(input_dim=34, hidden_dim=16, output_dim=embed_dim, num_layers=2)
        self.enc_voice = Conv1DEncoder(input_dim=24, hidden_dim=16, output_dim=embed_dim, num_layers=2)
        self.enc_physio= Conv1DEncoder(input_dim=14, hidden_dim=8,  output_dim=embed_dim, num_layers=1)
        
        # Stage 2: MoE fusion
        self.moe = MoEFusion(input_dim=3*embed_dim, num_experts=4, hidden_dim=hidden_dim, output_dim=embed_dim)
        
        # Stage 3: Output heads
        self.stress_head     = nn.Linear(embed_dim, 2)
        self.confidence_head = nn.Linear(embed_dim, 1)
        self.subj_head       = nn.Linear(embed_dim, num_subjects)
        
    def forward(self, face, voice, physio, return_all=False, return_confidence=False):
        # Stage 1
        ef = self.enc_face(face)     # [B, 8]
        ev = self.enc_voice(voice)   # [B, 8]
        ep = self.enc_physio(physio) # [B, 8]
        
        # Concatenate for MoE
        cat = torch.cat([ef, ev, ep], dim=1)  # [B, 24]
        
        # Stage 2
        fused, gate_weights = self.moe(cat)   # [B, 8]
        
        # Stage 3
        stress_logits = self.stress_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)
        
        rev_fused = grad_reverse(fused)        # GRL
        subj_logits = self.subj_head(rev_fused)
        
        if return_confidence:
            return stress_logits, subj_logits, confidence
        if return_all:
            return {
                "stress_logits":   stress_logits,
                "confidence":      confidence,
                "subj_logits":     subj_logits,
                "gate_weights":    gate_weights,
                "fused_embedding": fused,
                "modality_embeddings": {"face": ef, "voice": ev, "physio": ep},
            }
        return stress_logits, confidence
```

---

## Verification Plan

| Test | Expected Result |
|------|----------------|
| **Parameter count** | ~12,000 (verify with `sum(p.numel() for p in model.parameters())`) |
| **LOSO 5-fold** | Accuracy > current VBC-CASA-IS baseline |
| **ECE (calibration)** | < 15% (vs 52.57% on VBC-CASA-IS) |
| **Training time** | ~5min per fold (vs current ~30min+) |
| **MoE gate entropy** | Gates should spread weight across 3 modalities, not collapse to 1 |

---

## Training Procedure

Same as `train_ssvb_production.py` but with ConvMoE-MF model class:

```
1. Load enriched data (89K windows, 91 subjects)
2. StandardScaler per modality → z-score normalize
3. LOSO 5-fold split (no subject overlap)
4. Train each fold: Adam, LR=1e-3, batch=64, epochs=50
5. Loss: CE(stress) + 0.1×CE(subject_grl) + 0.1×BCE(confidence)
6. Evaluate: accuracy, precision, recall, F1, ECE, gate entropy
```

---

## Hybrid Architecture Note

If CNN-only proves insufficient for long-range temporal patterns in future (N > 10K), the MoE fusion module can be swapped for cross-attention without changing Stage 1 or Stage 3 — the [8] embedding interface stays the same.
