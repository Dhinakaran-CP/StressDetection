"""
ConvMoE-MF: Convolutional Mixture of Experts for Multimodal Fusion.

Architecture (3 stages, ~10K params):
  Stage 1: Per-modality Conv1D encoders with GAP (Face→8, Voice→8, Physio→8)
  Stage 2: MoE fusion (4 experts × 24→8, router 24→4)
  Stage 3: Stress, Confidence, Adversarial Subject heads

Drop-in compatible with SSVBCASA_AIS forward signature (9 sub-modality tensors).
"""
import numpy as np
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.02):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=0.02):
    return GradientReversal.apply(x, alpha)


class Conv1DEncoder(nn.Module):
    """Per-modality 1D-CNN encoder with global average pooling.

    Args:
        input_dim: Number of input feature channels
        hidden_dim: Hidden channel count (first conv layer)
        output_dim: Output embedding dimension (second conv + GAP)
        num_layers: 1 or 2 conv layers
        kernel_size: Conv kernel (default 5 for rich modalities, 3 for small)
    """
    def __init__(self, input_dim, hidden_dim=16, output_dim=8, num_layers=2, kernel_size=None):
        super().__init__()
        self.num_layers = num_layers
        if kernel_size is None:
            kernel_size = 5 if input_dim > 20 else 3
        ks = kernel_size

        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
            ks = 3  # second layer uses smaller kernel

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.mean(dim=2)
        return x


class MoEFusion(nn.Module):
    """Compact Mixture-of-Experts fusion.

    Args:
        input_dim: Concatenated modality embedding dimension (default 24)
        num_experts: Number of experts (default 4)
        hidden_dim: Expert hidden layer size
        output_dim: Output fused embedding dimension
    """
    def __init__(self, input_dim=24, num_experts=4, hidden_dim=16, output_dim=8):
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

    def forward(self, x):
        weights = self.router(x)
        outputs = torch.stack([e(x) for e in self.experts], dim=1)
        fused = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        return fused, weights


class ConvMoE_MF(nn.Module):
    """ConvMoE-MF: Convolutional Mixture of Experts for Multimodal Fusion.

    Drop-in replacement for SSVBCASA_AIS. Accepts the same 9 sub-modality
    tensors, internally concatenates into 3 modalities (face/voice/physio),
    passes through light Conv1D encoders, fuses via MoE, and produces
    stress + confidence + subject-adversarial outputs.

    Total params: ~10K (40× fewer than SSVB-CASA-AIS).
    """
    def __init__(self, hidden_dim=16, embed_dim=8, num_subjects=65,
                 face_dim=33, voice_dim=23, physio_dim=13):
        super().__init__()
        self.embed_dim = embed_dim

        self.enc_face  = Conv1DEncoder(input_dim=face_dim,  hidden_dim=16, output_dim=embed_dim, num_layers=2)
        self.enc_voice = Conv1DEncoder(input_dim=voice_dim, hidden_dim=16, output_dim=embed_dim, num_layers=2)
        self.enc_physio= Conv1DEncoder(input_dim=physio_dim, hidden_dim=8,  output_dim=embed_dim, num_layers=1)

        self.moe = MoEFusion(input_dim=3*embed_dim, num_experts=4, hidden_dim=hidden_dim, output_dim=embed_dim)

        self.stress_head     = nn.Linear(embed_dim, 2)
        self.confidence_head = nn.Linear(embed_dim, 1)
        self.subj_head       = nn.Linear(embed_dim, num_subjects)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.confidence_head.bias, 0.5)

    def _combine_modalities(self, eye, mouth, global_face,
                             spectral_prosody, mfcc, quality,
                             cardio, eda, somatic):
        """Combine 9 sub-modality tensors into 3 modality tensors."""
        face   = torch.cat([eye, mouth, global_face], dim=-1)
        voice  = torch.cat([spectral_prosody, mfcc, quality], dim=-1)
        physio = torch.cat([cardio, eda, somatic], dim=-1)
        return face, voice, physio

    def forward(self, eye, mouth, global_face,
                spectral_prosody, mfcc, quality,
                cardio, eda, somatic,
                return_all=False, return_confidence=False,
                quality_masks=None):
        face, voice, physio = self._combine_modalities(
            eye, mouth, global_face, spectral_prosody, mfcc, quality, cardio, eda, somatic)

        ef = self.enc_face(face)
        ev = self.enc_voice(voice)
        ep = self.enc_physio(physio)

        cat = torch.cat([ef, ev, ep], dim=1)
        fused, gate_weights = self.moe(cat)

        stress_logits = self.stress_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)

        rev_fused = grad_reverse(fused)
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

    def forward_from_latents(self, latent_f, latent_v, latent_p,
                             quality_masks=None, return_all=False,
                             return_confidence=False):
        """Bypass encoders, run MoE → heads on modality latents directly."""
        cat = torch.cat([latent_f, latent_v, latent_p], dim=1)
        fused, gate_weights = self.moe(cat)

        stress_logits = self.stress_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)

        rev_fused = grad_reverse(fused)
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
            }
        return stress_logits, confidence


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = ConvMoE_MF(num_subjects=65)
    print(f"ConvMoE-MF params: {count_params(m):,}")
    B, T = 4, 30
    dummy = {
        "eye":              torch.randn(B, T, 9),
        "mouth":            torch.randn(B, T, 6),
        "global_face":      torch.randn(B, T, 18),
        "spectral_prosody": torch.randn(B, T, 8),
        "mfcc":             torch.randn(B, T, 13),
        "quality":          torch.randn(B, T, 2),
        "cardio":           torch.randn(B, T, 2),
        "eda":              torch.randn(B, T, 3),
        "somatic":          torch.randn(B, T, 8),
    }
    out = m(**dummy, return_all=True)
    print(f"Stress logits: {out['stress_logits'].shape}")
    print(f"Confidence:    {out['confidence'].shape}")
    print(f"Subject logits:{out['subj_logits'].shape}")
    print(f"Gate weights:  {out['gate_weights'].shape}")
    print(f"Fused emb:     {out['fused_embedding'].shape}")
    print("Drop-in compatible with SSVBCASA_AIS OK")
