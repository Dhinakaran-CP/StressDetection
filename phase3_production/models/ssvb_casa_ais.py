"""
SSVB-CASA-AIS: Hybrid Mixture-of-Experts with Cross-Attention and
               Adversarial Identity Suppression.

Matches the full research champion architecture from
research/Phase_2_High_Capacity/models.py :: HybridMoEAttentionModel.

Architecture (6 stages):
  Stage 1: 8 raw sub-modality streams
  Stage 2: 8 SequenceEncoder sub-experts (one per sub-modality)
  Stage 3: 3 Intra-Modality Gated Routers (Face/Video/Physio Gates)
  Stage 4: 6 Inter-Modality Cross-Attention blocks + quality masking
  Stage 5: Global MoE Router + Sequence Mean Pooling
  Stage 6: 3 Output Heads (Stress, Confidence, Adversarial Subject)

All new layers are pass-through initialised so existing adv_ encoder
predictions are preserved at deployment.
"""
import numpy as np
import torch
import torch.nn as nn


# ── Stage 6: Gradient Reversal (GRL) ─────────────────────────────────────
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


# ── Stage 2: Sub-Modality Sequence Expert (1D-CNN + GRU) ────────────────
class SequenceExpert(nn.Module):
    """Sub-modality encoder matching the research SequenceEncoder.

    Architecture: Conv1D → BN → ReLU → MultiheadSelfAttention (residual)
                 → GRU → last-timestep-pool → [batch, hidden_dim]

    Adapted to handle both single-frame (batch, feat_dim) and
    multi-frame (batch, seq_len, feat_dim) inputs."""
    def __init__(self, input_dim, hidden_dim=16, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                batch_first=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x, return_sequence=False):
        """Forward pass.

        Args:
            x: [batch, feat_dim] or [batch, seq_len, feat_dim]
            return_sequence: if True, return full [batch, seq_len, hidden_dim]
                             sequence (used during training for dual_rep).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        gru_out, _ = self.gru(x)
        if return_sequence:
            return gru_out
        return gru_out[:, -1, :]


# ── Stage 3: Intra-Modality Gate ─────────────────────────────────────────
class IntraModalityGate(nn.Module):
    """Gated fusion over sub-experts within a single modality.

    Takes a list of sub-modality latent vectors (one per sub-expert)
    and learns per-sub-modality routing weights."""
    def __init__(self, num_sub, embed_dim=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_sub * embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_sub),
            nn.Softmax(dim=1),
        )

    def forward(self, latents):
        cat_in = torch.cat(latents, dim=1)
        weights = self.gate_net(cat_in)
        fused = torch.zeros_like(latents[0])
        for i, emb in enumerate(latents):
            fused += weights[:, i:i+1] * emb
        return fused

    def _init_equal_weights(self):
        """Initialise to equal weighting across sub-experts."""
        nn.init.constant_(self.gate_net[2].weight, 1.0 / self.gate_net[2].in_features)
        nn.init.constant_(self.gate_net[2].bias, 0)


# ── Stage 4: Cross-Attention Block (single-head) ─────────────────────────
class CrossAttentionBlock(nn.Module):
    """Target attends to Source using dot-product cross-attention.

    Supports optional source_quality_mask; when a source modality's
    quality score < 0.5 the attention weight is driven toward zero
    (§5.8 — Quality-Aware Selective Attention)."""
    def __init__(self, embed_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_target, x_source, source_quality_mask=None):
        q = self.q_proj(x_target.unsqueeze(1))
        k = self.k_proj(x_source.unsqueeze(1))
        v = self.v_proj(x_source.unsqueeze(1))
        scores = torch.bmm(q, k.permute(0, 2, 1)) / np.sqrt(q.shape[-1])
        if source_quality_mask is not None:
            mask_val = (source_quality_mask < 0.5).float().unsqueeze(-1).unsqueeze(-1)
            scores = scores - mask_val * 1e9
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, v)
        return self.out_proj(context.squeeze(1))


# ── Stage 5: Global MoE Router ───────────────────────────────────────────
class GlobalMoERouter(nn.Module):
    """Learns per-modality gating coefficients over the three
    cross-attention-reinforced modality embeddings."""
    def __init__(self, num_modalities=3, embed_dim=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_modalities),
            nn.Softmax(dim=1),
        )

    def forward(self, embeddings_list):
        cat_in = torch.cat(embeddings_list, dim=1)
        weights = self.gate_net(cat_in)
        fused = torch.zeros_like(embeddings_list[0])
        for i, emb in enumerate(embeddings_list):
            fused += weights[:, i:i+1] * emb
        return fused, weights


# ========================================================================
# Full SSVB-CASA-AIS Model
# ========================================================================
class SSVBCASA_AIS(nn.Module):
    """
    Full 6-stage SSVB-CASA-AIS architecture matching
    HybridMoEAttentionModel from the research code.

    Inference path (pass-through init preserves existing adv_ predictions):
      8 sub-modality feature vectors → 8 SequenceExperts →
      3 IntraModalityGates → 6 Cross-Attention blocks →
      3 projection layers → GlobalMoERouter →
      Stress/Confidence/Subject output heads

    For deployment against the existing ModalityEncoder backbones,
    the caller can bypass Stages 2-3 by passing latents directly
    via forward_from_latents().
    """
    def __init__(self, hidden_dim=16, num_subjects=65):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Stage 2: Sub-modality experts ──
        # Face group (3 experts) — 33 features total
        self.exp_eye         = SequenceExpert(input_dim=9,  hidden_dim=hidden_dim)
        self.exp_mouth       = SequenceExpert(input_dim=6,  hidden_dim=hidden_dim)
        self.exp_global_face = SequenceExpert(input_dim=18, hidden_dim=hidden_dim)
        # Voice group (3 experts) — 23 features total
        self.exp_spectral_prosody = SequenceExpert(input_dim=8,  hidden_dim=hidden_dim)
        self.exp_mfcc             = SequenceExpert(input_dim=13, hidden_dim=hidden_dim)
        self.exp_quality          = SequenceExpert(input_dim=2,  hidden_dim=hidden_dim)
        # Physio group (3 experts) — 13 features total
        self.exp_cardio  = SequenceExpert(input_dim=2, hidden_dim=hidden_dim)
        self.exp_eda     = SequenceExpert(input_dim=3, hidden_dim=hidden_dim)
        self.exp_somatic = SequenceExpert(input_dim=8, hidden_dim=hidden_dim)

        # ── Stage 3: Intra-modality gates (3 sub-experts per modality) ──
        self.gate_face   = IntraModalityGate(num_sub=3, embed_dim=hidden_dim)
        self.gate_voice  = IntraModalityGate(num_sub=3, embed_dim=hidden_dim)
        self.gate_physio = IntraModalityGate(num_sub=3, embed_dim=hidden_dim)

        # ── Stage 4: Cross-attention blocks ──
        self.attn_fv = CrossAttentionBlock(hidden_dim)
        self.attn_fp = CrossAttentionBlock(hidden_dim)
        self.attn_vf = CrossAttentionBlock(hidden_dim)
        self.attn_vp = CrossAttentionBlock(hidden_dim)
        self.attn_pf = CrossAttentionBlock(hidden_dim)
        self.attn_pv = CrossAttentionBlock(hidden_dim)

        # Projection: concat(cross_attn_pair) → hidden_dim
        self.proj_f = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim * 2, hidden_dim)
        self.proj_p = nn.Linear(hidden_dim * 2, hidden_dim)

        # ── Stage 5: Global MoE Router ──
        self.global_moe = GlobalMoERouter(num_modalities=3, embed_dim=hidden_dim)

        # ── Stage 6: Output heads ──
        self.stress_head     = nn.Linear(hidden_dim, 2)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.subj_head       = nn.Linear(hidden_dim, num_subjects)

        self._init_pass_through()

    def _init_pass_through(self):
        """Pass-through initialisation so existing adv_ encoder predictions
        are preserved at deployment time:

        - Sub-experts: small random weights (not zero — need signal)
        - Intra-modality gates: equal weights
        - Cross-attention out_proj: zero (no attention residual at init)
        - Global MoE: equal weights across 3 modalities
        - Stress head: zero (followed by pass-through init override)

        The stress_head is initialised to zero because at deployment
        the fused logit is a weighted average of the individual modality
        logits; the SSVB stress head will be fine-tuned later.
        """
        for gate in [self.gate_face, self.gate_voice, self.gate_physio]:
            gate._init_equal_weights()

        for attn in [self.attn_fv, self.attn_fp, self.attn_vf,
                     self.attn_vp, self.attn_pf, self.attn_pv]:
            nn.init.zeros_(attn.out_proj.weight)
            nn.init.zeros_(attn.out_proj.bias)

        nn.init.constant_(self.global_moe.gate_net[2].weight, 1.0 / 3)
        nn.init.constant_(self.global_moe.gate_net[2].bias, 0)

        nn.init.zeros_(self.stress_head.weight)
        nn.init.zeros_(self.stress_head.bias)

        nn.init.constant_(self.confidence_head.bias, 0)
        nn.init.constant_(self.subj_head.weight, 1e-3)
        nn.init.zeros_(self.subj_head.bias)

    # ── Full forward (Stages 2-6) ────────────────────────────────────────
    def forward(self, eye, mouth, global_face,
                spectral_prosody, mfcc, quality,
                cardio, eda, somatic,
                quality_masks=None, return_all=False, return_confidence=False):
        """
        Full 6-stage forward pass from expanded 10 sub-modality feature vectors.

        Parameters
        ----------
        eye, mouth, global_face  : Tensor [batch, T, 9|6|18]  — Face (3 experts)
        spectral_prosody, mfcc, quality : Tensor [batch, T, 8|13|2] — Voice (3 experts)
        cardio, eda, somatic     : Tensor [batch, T, 2|3|8]   — Physio (3 experts)
        quality_masks : dict or None — quality scores per modality.
        return_all   : bool — return full intermediate dict.
        return_confidence : bool — return (stress_logits, subj_logits, confidence)
        """
        e_eye   = self.exp_eye(eye)
        e_mouth = self.exp_mouth(mouth)
        e_gface = self.exp_global_face(global_face)
        e_sp    = self.exp_spectral_prosody(spectral_prosody)
        e_mfcc  = self.exp_mfcc(mfcc)
        e_qual  = self.exp_quality(quality)
        e_card  = self.exp_cardio(cardio)
        e_eda   = self.exp_eda(eda)
        e_soma  = self.exp_somatic(somatic)

        ef = self.gate_face([e_eye, e_mouth, e_gface])
        ev = self.gate_voice([e_sp, e_mfcc, e_qual])
        ep = self.gate_physio([e_card, e_eda, e_soma])

        return self._forward_stages_4_6(ef, ev, ep,
                                        quality_masks=quality_masks,
                                        return_all=return_all,
                                        return_confidence=return_confidence)

    # ── Latent-mode forward (Stages 4-6 only) ────────────────────────────
    def forward_from_latents(self, latent_f, latent_v, latent_p,
                             quality_masks=None, return_all=False,
                             return_confidence=False):
        """Bypass Stages 2-3, run cross-attention → heads on modality latents."""
        return self._forward_stages_4_6(latent_f, latent_v, latent_p,
                                        quality_masks=quality_masks,
                                        return_all=return_all,
                                        return_confidence=return_confidence)

    def _forward_stages_4_6(self, ef, ev, ep,
                            quality_masks=None, return_all=False,
                            return_confidence=False):
        """Shared cross-attention → global MoE → heads logic."""
        qm_f = quality_masks.get("face")   if quality_masks else None
        qm_v = quality_masks.get("voice")  if quality_masks else None
        qm_p = quality_masks.get("physio") if quality_masks else None

        f_re = self.proj_f(torch.cat([
            self.attn_fv(ef, ev, source_quality_mask=qm_v),
            self.attn_fp(ef, ep, source_quality_mask=qm_p),
        ], dim=1))
        v_re = self.proj_v(torch.cat([
            self.attn_vf(ev, ef, source_quality_mask=qm_f),
            self.attn_vp(ev, ep, source_quality_mask=qm_p),
        ], dim=1))
        p_re = self.proj_p(torch.cat([
            self.attn_pf(ep, ef, source_quality_mask=qm_f),
            self.attn_pv(ep, ev, source_quality_mask=qm_v),
        ], dim=1))

        fused, gate_weights = self.global_moe([f_re, v_re, p_re])

        stress_logits = self.stress_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)

        # Subject head with GRL (always available at inference, used during training)
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
                "f_re": f_re,
                "v_re": v_re,
                "p_re": p_re,
            }
        return stress_logits, confidence


# ── Feature Sub-Modality Split ───────────────────────────────────────────
SUB_MODALITY_INDICES = {
    "face": {
        "eye":         [0, 1, 2, 3, 15],
        "mouth":       [7, 8, 9],
        "global_face": [4, 5, 6, 10, 12, 13, 14, 17],
    },
    "voice": {
        "prosody":     [10, 11],
        "spectral":    [8, 9],
        "quality":     [1, 3, 4, 5, 6, 7],   # ADDED f0_std (idx 1), was [3,4,5,6,7]
    },
    "physio": {
        "cardio":      [0, 1, 2],
        "motion":      [4],
    },
}


def split_sub_modalities(modality, flat_features):
    """Split a flat [n,] feature vector into sub-modality dict for SSVB-CASA-AIS."""
    import numpy as np
    if flat_features is None:
        return None
    groups = SUB_MODALITY_INDICES.get(modality, {})
    return {
        name: np.take(flat_features, indices).tolist()
        for name, indices in groups.items()
    }
