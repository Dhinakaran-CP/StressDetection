import os
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------
# Gradient Reversal Layer for Adversarial Suppression
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Base 1D-CNN + GRU Sequence Encoder (Unimodal Expert)
# ---------------------------------------------------------
class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_heads=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x, return_sequence=False):
        # Input shape: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]
        
        # Standard multi-head self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out  # residual connection
        
        gru_out, _ = self.gru(x)
        if return_sequence:
            return gru_out
        latent = gru_out[:, -1, :]  # Last time step [batch, hidden_dim]
        return latent

# ---------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x_target, x_source):
        # Inputs shape: [batch, embed_dim]
        # Unsqueeze to add a pseudo sequence dimension [batch, 1, embed_dim]
        q = self.q_proj(x_target.unsqueeze(1))
        k = self.k_proj(x_source.unsqueeze(1))
        v = self.v_proj(x_source.unsqueeze(1))
        
        scores = torch.bmm(q, k.permute(0, 2, 1)) / np.sqrt(q.shape[-1])  # [batch, 1, 1]
        weights = torch.softmax(scores, dim=-1)
        
        context = torch.bmm(weights, v)  # [batch, 1, embed_dim]
        out = self.out_proj(context.squeeze(1))  # [batch, embed_dim]
        return out

# ---------------------------------------------------------
# Multi-Head Cross-Attention Block (VBC-CASA-IS)
# ---------------------------------------------------------
class MultiheadCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_target, x_source, source_quality_mask=None):
        """
        Parameters
        ----------
        x_target            : [batch, seq_len, embed_dim]
        x_source            : [batch, seq_len, embed_dim]
        source_quality_mask : [batch] float tensor in [0,1] — quality score of source modality.
                              When provided, positions where quality == 0.0 are masked out
                              so the attention head ignores the low-quality source stream.
        """
        # Build key_padding_mask: True = ignore that key position
        key_padding_mask = None
        if source_quality_mask is not None:
            # source_quality_mask shape: [batch]
            # Broadcast to [batch, seq_len]: mask entire sequence if quality is zero
            bad = (source_quality_mask < 0.5)                           # [batch]
            key_padding_mask = bad.unsqueeze(1).expand(                 # [batch, seq_len]
                -1, x_source.size(1)
            )

        attn_out, _ = self.attn(
            query=x_target,
            key=x_source,
            value=x_source,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(x_target + attn_out)

# ---------------------------------------------------------
# Gated Fusion Layer
# ---------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, num_modalities=3, embed_dim=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_modalities),
            nn.Softmax(dim=1)
        )
        
    def forward(self, embeddings_list):
        # Concatenate embeddings to calculate gating coefficients
        cat_in = torch.cat(embeddings_list, dim=1)
        weights = self.gate_net(cat_in)  # [batch, num_modalities]
        
        fused = torch.zeros_like(embeddings_list[0])
        for idx, emb in enumerate(embeddings_list):
            fused += weights[:, idx : idx + 1] * emb
        return fused

# ---------------------------------------------------------
# Sequence-level Gated Fusion (VBC-CASA-IS)
# ---------------------------------------------------------
class SequenceGatedFusion(nn.Module):
    def __init__(self, num_modalities=3, embed_dim=16):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_modalities),
            nn.Softmax(dim=1)
        )
        
    def forward(self, embeddings_list):
        # Embeddings list elements: [batch, seq_len, embed_dim]
        # Pool temporally (mean) to calculate static weights
        pooled = [e.mean(dim=1) for e in embeddings_list]
        cat_in = torch.cat(pooled, dim=1)  # [batch, num_modalities * embed_dim]
        weights = self.gate_net(cat_in)     # [batch, num_modalities]
        
        fused = torch.zeros_like(embeddings_list[0])
        for idx, emb in enumerate(embeddings_list):
            # Scale along sequence steps
            fused += weights[:, idx : idx + 1].unsqueeze(-1) * emb
        return fused

# =========================================================
# Stage 1: Unimodal Modality Experts
# =========================================================
class UnimodalExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, num_subjects=65, adversarial=False):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden_dim)
        self.stress_head = nn.Linear(hidden_dim, 2)
        
        self.adversarial = adversarial
        if adversarial:
            self.subj_head = nn.Linear(hidden_dim, num_subjects)
            
    def forward(self, x):
        latent = self.encoder(x)
        stress_logits = self.stress_head(latent)
        
        if self.adversarial:
            rev_latent = grad_reverse(latent)
            subj_logits = self.subj_head(rev_latent)
            return stress_logits, subj_logits
            
        return stress_logits

# =========================================================
# Stage 2 & 3: Multi-modal Gated & Early Fusion Models
# =========================================================
class EarlyFusionModel(nn.Module):
    def __init__(self, face_dim, voice_dim, physio_dim, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceEncoder(face_dim, hidden_dim)
        self.enc_v = SequenceEncoder(voice_dim, hidden_dim)
        self.enc_p = SequenceEncoder(physio_dim, hidden_dim)
        
        # Concat projection classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, face_x, voice_x, physio_x):
        ef = self.enc_f(face_x)
        ev = self.enc_v(voice_x)
        ep = self.enc_p(physio_x)
        
        cat_emb = torch.cat([ef, ev, ep], dim=1)
        return self.classifier(cat_emb)

class GatedFusionModel(nn.Module):
    def __init__(self, face_dim, voice_dim, physio_dim, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceEncoder(face_dim, hidden_dim)
        self.enc_v = SequenceEncoder(voice_dim, hidden_dim)
        self.enc_p = SequenceEncoder(physio_dim, hidden_dim)
        
        self.gated_fusion = GatedFusion(num_modalities=3, embed_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, face_x, voice_x, physio_x):
        ef = self.enc_f(face_x)
        ev = self.enc_v(voice_x)
        ep = self.enc_p(physio_x)
        
        fused = self.gated_fusion([ef, ev, ep])
        return self.classifier(fused)

# =========================================================
# Stage 4: Cross-Attention Multimodal Model
# =========================================================
class CrossAttentionFusionModel(nn.Module):
    def __init__(self, face_dim, voice_dim, physio_dim, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceEncoder(face_dim, hidden_dim)
        self.enc_v = SequenceEncoder(voice_dim, hidden_dim)
        self.enc_p = SequenceEncoder(physio_dim, hidden_dim)
        
        # Cross-Attention blocks
        self.attn_fv = CrossAttentionBlock(hidden_dim)
        self.attn_fp = CrossAttentionBlock(hidden_dim)
        self.attn_vf = CrossAttentionBlock(hidden_dim)
        self.attn_vp = CrossAttentionBlock(hidden_dim)
        self.attn_pf = CrossAttentionBlock(hidden_dim)
        self.attn_pv = CrossAttentionBlock(hidden_dim)
        
        # Projection layer
        self.proj_f = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_p = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.gated_fusion = GatedFusion(num_modalities=3, embed_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, face_x, voice_x, physio_x):
        ef = self.enc_f(face_x)
        ev = self.enc_v(voice_x)
        ep = self.enc_p(physio_x)
        
        # Cross-attention reinforcement
        f_reinforced = self.proj_f(torch.cat([self.attn_fv(ef, ev), self.attn_fp(ef, ep)], dim=1))
        v_reinforced = self.proj_v(torch.cat([self.attn_vf(ev, ef), self.attn_vp(ev, ep)], dim=1))
        p_reinforced = self.proj_p(torch.cat([self.attn_pf(ep, ef), self.attn_pv(ep, ev)], dim=1))
        
        fused = self.gated_fusion([f_reinforced, v_reinforced, p_reinforced])
        return self.classifier(fused)

# =========================================================
# Stage 5 & 6: Specialized Sub-Modality MoE & Hybrid Model
# =========================================================
class HybridMoEAttentionModel(nn.Module):
    def __init__(self, hidden_dim=16, num_subjects=65, adversarial=False, dual_representation=False):
        super().__init__()
        self.adversarial = adversarial
        self.dual_representation = dual_representation
        
        # Sub-modality specialized experts (Adjust input dimensions based on dual_representation)
        factor = 2 if self.dual_representation else 1
        self.exp_eye = SequenceEncoder(input_dim=5 * factor, hidden_dim=hidden_dim)
        self.exp_mouth = SequenceEncoder(input_dim=3 * factor, hidden_dim=hidden_dim)
        self.exp_global_face = SequenceEncoder(input_dim=8 * factor, hidden_dim=hidden_dim)
        
        self.exp_prosody = SequenceEncoder(input_dim=3 * factor, hidden_dim=hidden_dim)
        self.exp_spectral = SequenceEncoder(input_dim=2 * factor, hidden_dim=hidden_dim)
        self.exp_quality = SequenceEncoder(input_dim=5 * factor, hidden_dim=hidden_dim)
        
        self.exp_cardio = SequenceEncoder(input_dim=3 * factor, hidden_dim=hidden_dim)
        self.exp_motion = SequenceEncoder(input_dim=1 * factor, hidden_dim=hidden_dim)
        
        # Intra-modality internal gating routers
        if self.dual_representation:
            self.gate_face = SequenceGatedFusion(num_modalities=3, embed_dim=hidden_dim)
            self.gate_voice = SequenceGatedFusion(num_modalities=3, embed_dim=hidden_dim)
            self.gate_physio = SequenceGatedFusion(num_modalities=2, embed_dim=hidden_dim)
            
            # Cross-Attention blocks (Multi-Head Cross-Attention)
            self.attn_fv = MultiheadCrossAttentionBlock(hidden_dim)
            self.attn_fp = MultiheadCrossAttentionBlock(hidden_dim)
            self.attn_vf = MultiheadCrossAttentionBlock(hidden_dim)
            self.attn_vp = MultiheadCrossAttentionBlock(hidden_dim)
            self.attn_pf = MultiheadCrossAttentionBlock(hidden_dim)
            self.attn_pv = MultiheadCrossAttentionBlock(hidden_dim)
            
            # Global MoE Router
            self.global_gate = SequenceGatedFusion(num_modalities=3, embed_dim=hidden_dim)
        else:
            self.gate_face = GatedFusion(num_modalities=3, embed_dim=hidden_dim)
            self.gate_voice = GatedFusion(num_modalities=3, embed_dim=hidden_dim)
            self.gate_physio = GatedFusion(num_modalities=2, embed_dim=hidden_dim)
            
            # Cross-Attention blocks
            self.attn_fv = CrossAttentionBlock(hidden_dim)
            self.attn_fp = CrossAttentionBlock(hidden_dim)
            self.attn_vf = CrossAttentionBlock(hidden_dim)
            self.attn_vp = CrossAttentionBlock(hidden_dim)
            self.attn_pf = CrossAttentionBlock(hidden_dim)
            self.attn_pv = CrossAttentionBlock(hidden_dim)
            
            # Global MoE Router
            self.global_gate = GatedFusion(num_modalities=3, embed_dim=hidden_dim)
            
        self.proj_f = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_p = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.stress_head = nn.Linear(hidden_dim, 2)
        self.confidence_head = nn.Linear(hidden_dim, 1) # Auxiliary confidence head
        
        if adversarial:
            self.subj_head = nn.Linear(hidden_dim, num_subjects)
            
    def forward(self, eye, mouth, global_face, prosody, spectral, quality, cardio, motion,
                return_confidence=False, quality_masks=None):
        """
        Parameters
        ----------
        eye … motion       : modality tensors [batch, seq_len, feat_dim]
        return_confidence  : bool
        quality_masks      : dict {"face": Tensor[batch], "voice": Tensor[batch], "physio": Tensor[batch]}
                             Optional per-sample quality scores in [0,1] for each modality.
                             Passed as source_quality_mask into MultiheadCrossAttentionBlock
                             so low-quality streams are masked (§5.8).
        """
        # Unpack quality masks (default: None → no masking)
        qm_f = quality_masks["face"]   if (quality_masks and self.dual_representation) else None
        qm_v = quality_masks["voice"]  if (quality_masks and self.dual_representation) else None
        qm_p = quality_masks["physio"] if (quality_masks and self.dual_representation) else None

        # 1. Forward through specialized sub-experts
        e_eye = self.exp_eye(eye, return_sequence=self.dual_representation)
        e_mouth = self.exp_mouth(mouth, return_sequence=self.dual_representation)
        e_gface = self.exp_global_face(global_face, return_sequence=self.dual_representation)

        e_prosody = self.exp_prosody(prosody, return_sequence=self.dual_representation)
        e_spectral = self.exp_spectral(spectral, return_sequence=self.dual_representation)
        e_quality = self.exp_quality(quality, return_sequence=self.dual_representation)

        e_cardio = self.exp_cardio(cardio, return_sequence=self.dual_representation)
        e_motion = self.exp_motion(motion, return_sequence=self.dual_representation)

        # 2. Intra-modality aggregation
        ef = self.gate_face([e_eye, e_mouth, e_gface])
        ev = self.gate_voice([e_prosody, e_spectral, e_quality])
        ep = self.gate_physio([e_cardio, e_motion])

        # 3. Inter-modality cross-attention reinforcement
        if self.dual_representation:
            # MultiheadCrossAttentionBlock returns [batch, seq_len, hidden_dim]
            # Pass quality masks so low-quality source streams are ignored (§5.8)
            f_re = self.proj_f(torch.cat([
                self.attn_fv(ef, ev, source_quality_mask=qm_v),
                self.attn_fp(ef, ep, source_quality_mask=qm_p),
            ], dim=2))
            v_re = self.proj_v(torch.cat([
                self.attn_vf(ev, ef, source_quality_mask=qm_f),
                self.attn_vp(ev, ep, source_quality_mask=qm_p),
            ], dim=2))
            p_re = self.proj_p(torch.cat([
                self.attn_pf(ep, ef, source_quality_mask=qm_f),
                self.attn_pv(ep, ev, source_quality_mask=qm_v),
            ], dim=2))

            # 4. Global MoE routing (returns sequence [batch, seq_len, hidden_dim])
            fused = self.global_gate([f_re, v_re, p_re])

            # Save cross-attention representations for attention loss computation in training loop
            self.current_cross_outputs = {
                "f_re": f_re,
                "v_re": v_re,
                "p_re": p_re
            }

            # Internal sequence pooling before final heads
            fused_pooled = fused.mean(dim=1)
        else:
            f_re = self.proj_f(torch.cat([self.attn_fv(ef, ev), self.attn_fp(ef, ep)], dim=1))
            v_re = self.proj_v(torch.cat([self.attn_vf(ev, ef), self.attn_vp(ev, ep)], dim=1))
            p_re = self.proj_p(torch.cat([self.attn_pf(ep, ef), self.attn_pv(ep, ev)], dim=1))
            fused = self.global_gate([f_re, v_re, p_re])
            fused_pooled = fused

        stress_logits = self.stress_head(fused_pooled)

        if return_confidence:
            confidence = torch.sigmoid(self.confidence_head(fused_pooled))
            if self.adversarial:
                rev_fused = grad_reverse(fused_pooled)
                subj_logits = self.subj_head(rev_fused)
                return stress_logits, subj_logits, confidence
            return stress_logits, confidence

        if self.adversarial:
            rev_fused = grad_reverse(fused_pooled)
            subj_logits = self.subj_head(rev_fused)
            return stress_logits, subj_logits

        return stress_logits

