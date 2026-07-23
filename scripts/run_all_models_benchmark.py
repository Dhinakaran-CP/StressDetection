"""
Unified Research Model Benchmark.

Trains all research model architectures on the enriched data pipeline
(StressID -> WESAD -> Combined), using proper LOSO with fold purity fix,
single-class subject exclusion, per-dataset metrics, and GPU.

Usage:
    python scripts/run_all_models_benchmark.py
    python scripts/run_all_models_benchmark.py --models ssvb,cnn_baseline
    python scripts/run_all_models_benchmark.py --skip temporal
    python scripts/run_all_models_benchmark.py --list
    python scripts/run_all_models_benchmark.py --dry-run
"""
import sys, os, json, time, warnings, copy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase3_production'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Phase 3 / production imports ──────────────────────────────────────────
# train.py has __main__ guard so importing doesn't run main()
from phase3_production.train import (SSVBDataset, CNNBaseline, CNNBaselineGRL,
    CONFIG as TRAIN_CONFIG, _unpack_batch, calculate_metrics,
    find_optimal_threshold, per_subject_metrics, per_dataset_metrics,
    per_source_dataset_metrics, contrastive_loss, generate_plots)
from models.ssvb_casa_ais import SSVBCASA_AIS, SequenceExpert, CrossAttentionBlock
from models.conv_moe_mf import ConvMoE_MF, grad_reverse

ENRICHED_DIR = str(PROJECT_ROOT / 'data' / 'enriched_training_data')
REPORTS_DIR  = str(PROJECT_ROOT / 'benchmark_results')

SEED   = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Feature dimensions from enriched data (loaded at runtime) ─────────────
# Populated by first SSVBDataset construction
GROUP_KEYS = ['face_eye', 'face_global_face', 'face_mouth',
              'physio_cardio', 'physio_eda', 'physio_somatic',
              'voice_mfcc', 'voice_quality', 'voice_spectral_prosody']

# ====================================================================
# ADAPTED RESEARCH MODEL ARCHITECTURES
# (enriched-data-compatible versions preserving original design patterns)
# ====================================================================

# ── Gradient Reversal (used by adapted models) ─────────────────────
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=0.02):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_rev(x, alpha=0.02):
    return GradReverse.apply(x, alpha)

# ── Gated Fusion (from research/Phase_2_High_Capacity/models.py) ──
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
        cat_in = torch.cat(embeddings_list, dim=1)
        weights = self.gate_net(cat_in)
        fused = torch.zeros_like(embeddings_list[0])
        for idx, emb in enumerate(embeddings_list):
            fused += weights[:, idx:idx+1] * emb
        return fused

# ── Early Fusion Model (Phase 2) ───────────────────────────────────
class EarlyFusionModel(nn.Module):
    """Concatenates face/voice/physio encodings -> linear classifier."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceExpert(face_dim, hidden_dim)
        self.enc_v = SequenceExpert(voice_dim, hidden_dim)
        self.enc_p = SequenceExpert(physio_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, face, voice, physio):
        return self.classifier(torch.cat(
            [self.enc_f(face), self.enc_v(voice), self.enc_p(physio)], dim=1))

# ── Gated Fusion Model (Phase 2) ───────────────────────────────────
class GatedFusionModel(nn.Module):
    """Gated weighted fusion of modality encodings."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceExpert(face_dim, hidden_dim)
        self.enc_v = SequenceExpert(voice_dim, hidden_dim)
        self.enc_p = SequenceExpert(physio_dim, hidden_dim)
        self.gate = GatedFusion(3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, face, voice, physio):
        fused = self.gate([self.enc_f(face), self.enc_v(voice), self.enc_p(physio)])
        return self.classifier(fused)

# ── Cross-Attention Fusion Model (Phase 2) ─────────────────────────
class CrossAttentionFusionModel(nn.Module):
    """Cross-attention reinforced modality fusion."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, hidden_dim=16):
        super().__init__()
        self.enc_f = SequenceExpert(face_dim, hidden_dim)
        self.enc_v = SequenceExpert(voice_dim, hidden_dim)
        self.enc_p = SequenceExpert(physio_dim, hidden_dim)
        self.attn_fv = CrossAttentionBlock(hidden_dim)
        self.attn_fp = CrossAttentionBlock(hidden_dim)
        self.attn_vf = CrossAttentionBlock(hidden_dim)
        self.attn_vp = CrossAttentionBlock(hidden_dim)
        self.attn_pf = CrossAttentionBlock(hidden_dim)
        self.attn_pv = CrossAttentionBlock(hidden_dim)
        self.proj_f = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_p = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate = GatedFusion(3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, face, voice, physio):
        ef, ev, ep = self.enc_f(face), self.enc_v(voice), self.enc_p(physio)
        f_re = self.proj_f(torch.cat([self.attn_fv(ef, ev), self.attn_fp(ef, ep)], dim=1))
        v_re = self.proj_v(torch.cat([self.attn_vf(ev, ef), self.attn_vp(ev, ep)], dim=1))
        p_re = self.proj_p(torch.cat([self.attn_pf(ep, ef), self.attn_pv(ep, ev)], dim=1))
        return self.classifier(self.gate([f_re, v_re, p_re]))

# ── Adapted HybridMoEAttentionModel (Phase 2, enriched-compatible) ─
class HybridMoEAttentionModel(nn.Module):
    """Hybrid MoE + Cross-Attention (enriched-adapted: 9 groups -> 8 experts)."""
    def __init__(self, hidden_dim=16, num_subjects=91, adversarial=False):
        super().__init__()
        self.adversarial = adversarial
        self.exp_eye    = SequenceExpert(9, hidden_dim)
        self.exp_mouth  = SequenceExpert(6, hidden_dim)
        self.exp_gface  = SequenceExpert(18, hidden_dim)
        self.exp_prosody = SequenceExpert(8, hidden_dim)
        self.exp_spectral = SequenceExpert(13, hidden_dim)
        self.exp_quality = SequenceExpert(2, hidden_dim)
        self.exp_cardio = SequenceExpert(2, hidden_dim)
        self.exp_eda_soma = SequenceExpert(11, hidden_dim)
        self.gate_face   = GatedFusion(3, hidden_dim)
        self.gate_voice  = GatedFusion(3, hidden_dim)
        self.gate_physio = GatedFusion(2, hidden_dim)
        self.attn_fv = CrossAttentionBlock(hidden_dim)
        self.attn_fp = CrossAttentionBlock(hidden_dim)
        self.attn_vf = CrossAttentionBlock(hidden_dim)
        self.attn_vp = CrossAttentionBlock(hidden_dim)
        self.attn_pf = CrossAttentionBlock(hidden_dim)
        self.attn_pv = CrossAttentionBlock(hidden_dim)
        self.proj_f = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(2 * hidden_dim, hidden_dim)
        self.proj_p = nn.Linear(2 * hidden_dim, hidden_dim)
        self.global_gate = GatedFusion(3, hidden_dim)
        self.stress_head = nn.Linear(hidden_dim, 2)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        if adversarial:
            self.subj_head = nn.Linear(hidden_dim, num_subjects)
    def forward(self, eye, mouth, gface, sp, mfcc, qual, card, eda, soma,
                return_confidence=False):
        e_eye  = self.exp_eye(eye)
        e_mouth = self.exp_mouth(mouth)
        e_gface = self.exp_gface(gface)
        e_prosody = self.exp_prosody(sp)
        e_spectral = self.exp_spectral(mfcc)
        e_quality = self.exp_quality(qual)
        e_cardio = self.exp_cardio(card)
        e_eda_soma = self.exp_eda_soma(torch.cat([eda, soma], dim=-1))
        ef = self.gate_face([e_eye, e_mouth, e_gface])
        ev = self.gate_voice([e_prosody, e_spectral, e_quality])
        ep = self.gate_physio([e_cardio, e_eda_soma])
        f_re = self.proj_f(torch.cat([self.attn_fv(ef, ev), self.attn_fp(ef, ep)], dim=1))
        v_re = self.proj_v(torch.cat([self.attn_vf(ev, ef), self.attn_vp(ev, ep)], dim=1))
        p_re = self.proj_p(torch.cat([self.attn_pf(ep, ef), self.attn_pv(ep, ev)], dim=1))
        fused = self.global_gate([f_re, v_re, p_re])
        logits = self.stress_head(fused)
        if return_confidence:
            conf = torch.sigmoid(self.confidence_head(fused)).squeeze(-1)
            if self.adversarial:
                rev = grad_rev(fused)
                subj_l = self.subj_head(rev)
                return logits, subj_l, conf
            return logits, conf
        return logits

# ── Temporal Deep Models (Phase 4/5) ───────────────────────────────
class TemporalGRU(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TemporalTCN(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(input_dim, hidden_dim, 1) if input_dim != hidden_dim else nn.Identity()
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        res = self.proj(x)
        x = self.drop1(self.relu(self.bn1(self.conv1(x))))[:, :, :res.size(2)]
        x = self.drop2(self.relu(self.bn2(self.conv2(x))))[:, :, :res.size(2)]
        return self.fc(self.relu(x + res).mean(dim=2))

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=69, hidden_dim=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 30, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = self.proj(x) + self.pos_emb[:, :x.size(1), :]
        return self.fc(self.transformer(x).mean(dim=1))

# ── Expert Pipeline Model (Phase 6, adapted for 9 -> 8 subparts) ──
class SubpartExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :]))

class GatingRouter(nn.Module):
    def __init__(self, context_dim, num_experts=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, num_experts))
    def forward(self, context):
        return torch.softmax(self.net(context), dim=1)

class ExpertPipelineModel(nn.Module):
    """8 sub-part experts with gating router (9 enriched groups -> 8 subparts)."""
    def __init__(self, subpart_dims=None, hidden_dim=16):
        super().__init__()
        if subpart_dims is None:
            subpart_dims = [9, 6, 18, 8, 13, 2, 2, 11]
        self.experts = nn.ModuleList([SubpartExpert(d, hidden_dim) for d in subpart_dims])
        context_dim = sum(subpart_dims)
        self.router = GatingRouter(context_dim, len(subpart_dims))
    def forward(self, subpart_inputs):
        expert_l = [e(x) for e, x in zip(self.experts, subpart_inputs)]
        stacked = torch.stack(expert_l, dim=1)
        context = torch.cat([x[:, -1, :] for x in subpart_inputs], dim=-1)
        weights = self.router(context)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights, expert_l

# ── ModalityEncoder (generic, used by Phase 2 early fusion) ────────
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x.permute(0, 2, 1)))).permute(0, 2, 1)
        out, _ = self.gru(x)
        return out[:, -1, :]

class EarlyFusionClassifier(nn.Module):
    """Simple concat fusion (Phase 2 early fusion pipeline)."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, latent_dim=16):
        super().__init__()
        self.f_enc = ModalityEncoder(face_dim, latent_dim)
        self.v_enc = ModalityEncoder(voice_dim, latent_dim)
        self.p_enc = ModalityEncoder(physio_dim, latent_dim)
        self.clf = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(latent_dim, 2))
    def forward(self, face, voice, physio):
        return self.clf(torch.cat([self.f_enc(face), self.v_enc(voice), self.p_enc(physio)], dim=1))

class GatedFusionClassifier(nn.Module):
    """Learned gated fusion (Phase 2 early fusion pipeline)."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, latent_dim=16):
        super().__init__()
        self.f_enc = ModalityEncoder(face_dim, latent_dim)
        self.v_enc = ModalityEncoder(voice_dim, latent_dim)
        self.p_enc = ModalityEncoder(physio_dim, latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 3), nn.Softmax(dim=1))
        self.clf = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(latent_dim, 2))
    def forward(self, face, voice, physio):
        ef, ev, ep = self.f_enc(face), self.v_enc(voice), self.p_enc(physio)
        w = self.gate(torch.cat([ef, ev, ep], dim=1))
        return self.clf(w[:, 0:1]*ef + w[:, 1:2]*ev + w[:, 2:3]*ep)

class CrossAttentionFusionClassifier(nn.Module):
    """Self-attention fusion (Phase 2 early fusion pipeline)."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, latent_dim=16):
        super().__init__()
        self.f_enc = ModalityEncoder(face_dim, latent_dim)
        self.v_enc = ModalityEncoder(voice_dim, latent_dim)
        self.p_enc = ModalityEncoder(physio_dim, latent_dim)
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)
        self.v = nn.Linear(latent_dim, latent_dim)
        self.clf = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(latent_dim, 2))
    def forward(self, face, voice, physio):
        stacked = torch.stack([self.f_enc(face), self.v_enc(voice), self.p_enc(physio)], dim=1)
        attn = torch.softmax(torch.bmm(self.q(stacked), self.k(stacked).transpose(1,2))
                             / np.sqrt(stacked.size(-1)), dim=-1)
        return self.clf(torch.bmm(attn, self.v(stacked)).view(stacked.size(0), -1))

class FlexiModalMoE(nn.Module):
    """Flexible missing-modality MoE (Phase 2 early fusion pipeline)."""
    def __init__(self, face_dim=33, voice_dim=23, physio_dim=13, latent_dim=16,
                 num_experts=4, top_k=2):
        super().__init__()
        self.f_enc = ModalityEncoder(face_dim, latent_dim)
        self.v_enc = ModalityEncoder(voice_dim, latent_dim)
        self.p_enc = ModalityEncoder(physio_dim, latent_dim)
        self.missing_f = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.missing_v = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.missing_p = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(3*latent_dim, 2*latent_dim), nn.ReLU(),
                          nn.Dropout(0.1), nn.Linear(2*latent_dim, 3*latent_dim))
            for _ in range(num_experts)])
        self.router = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, num_experts))
        self.clf = nn.Sequential(
            nn.Linear(3*latent_dim, latent_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(latent_dim, 2))
        self.top_k = min(top_k, num_experts)
    def forward(self, face, voice, physio):
        ef, ev, ep = self.f_enc(face), self.v_enc(voice), self.p_enc(physio)
        joint = torch.cat([ef, ev, ep], dim=1)
        logits = self.router(joint)
        probs = torch.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        mask = torch.zeros_like(probs).scatter_(1, topk_idx, 1.0)
        gw = (probs * mask) / (torch.sum(probs * mask, dim=1, keepdim=True) + 1e-8)
        expert_out = torch.stack([e(joint) for e in self.experts], dim=1)
        return self.clf(torch.sum(gw.unsqueeze(-1) * expert_out, dim=1))

# ====================================================================
# MODEL REGISTRY
# ====================================================================

@dataclass
class ModelEntry:
    name: str
    cls: nn.Module
    group: str = 'research'
    init_kwargs: Dict = field(default_factory=dict)
    input_format: str = 'nine_tensor'   # nine_tensor | three_tensor | fused | subparts
    returns: str = 'logits_confidence'  # logits_only | logits_confidence | logits_subj_confidence
    ssl_epochs: int = 4
    ft_epochs: int = 8
    learning_rate: float = 5e-4
    description: str = ''

REGISTRY: List[ModelEntry] = [
    # ═══ PHASE 3 PRODUCTION ═══
    ModelEntry('ssvb_casa_ais', SSVBCASA_AIS, group='phase3',
        init_kwargs={'hidden_dim': 16, 'num_subjects': 91},
        returns='logits_subj_confidence',
        description='Full SSVB-CASA-AIS: 9 experts, cross-attention, global MoE, GRL'),

    ModelEntry('conv_moe_mf', ConvMoE_MF, group='phase3',
        init_kwargs={'hidden_dim': 16, 'embed_dim': 8, 'num_subjects': 91, 'num_datasets': 3,
                     'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13,
                     'grl_alpha_subj': 0.02, 'grl_alpha_ds': 0.02},
        returns='logits_subj_confidence',
        description='ConvMoE-MF: light conv encoders, MoE fusion, dual GRL'),

    ModelEntry('cnn_baseline', CNNBaseline, group='phase3',
        init_kwargs={'total_feat_dim': 69, 'hidden_dims': [64, 32, 16], 'num_classes': 2, 'num_subjects': 91},
        returns='logits_subj_confidence',
        description='CNNBaseline: plain 1D-CNN, no GRL'),

    ModelEntry('cnn_baseline_grl', CNNBaselineGRL, group='phase3',
        init_kwargs={'total_feat_dim': 69, 'hidden_dims': [64, 32, 16], 'num_classes': 2, 'num_subjects': 91, 'grl_alpha': 0.02},
        returns='logits_subj_confidence',
        description='CNNBaseline+GRL: 1D-CNN with adversarial subject head'),

    # ═══ PHASE 2 — HIGH CAPACITY FUSION ═══
    ModelEntry('early_fusion', EarlyFusionModel, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'hidden_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Early concat fusion: concat(enc(f), enc(v), enc(p)) -> linear'),

    ModelEntry('gated_fusion', GatedFusionModel, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'hidden_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Gated weighted fusion of modality encodings'),

    ModelEntry('cross_attn_fusion', CrossAttentionFusionModel, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'hidden_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Cross-attention reinforced modality fusion (6 cross-attn blocks)'),

    ModelEntry('hybrid_moe_attention', HybridMoEAttentionModel, group='phase2',
        init_kwargs={'hidden_dim': 16, 'num_subjects': 91, 'adversarial': False},
        returns='logits_confidence',
        description='Adapted Hybrid MoE + Cross-Attention (8 experts, 3 modality gates)'),

    ModelEntry('hybrid_moe_attention_adv', HybridMoEAttentionModel, group='phase2',
        init_kwargs={'hidden_dim': 16, 'num_subjects': 91, 'adversarial': True},
        returns='logits_subj_confidence',
        description='Hybrid MoE + Cross-Attention with adversarial subject GRL'),

    # ═══ PHASE 2 — EARLY FUSION PIPELINE ═══
    ModelEntry('early_fusion_clf', EarlyFusionClassifier, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'latent_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Simple concat fusion (Conv1D+GRU encoders)'),

    ModelEntry('gated_fusion_clf', GatedFusionClassifier, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'latent_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Learned soft gate fusion'),

    ModelEntry('cross_attn_clf', CrossAttentionFusionClassifier, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'latent_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='Self-attention modality fusion (Q/K/V over modalities)'),

    ModelEntry('fleximodal_moe', FlexiModalMoE, group='phase2',
        init_kwargs={'face_dim': 33, 'voice_dim': 23, 'physio_dim': 13, 'latent_dim': 16},
        input_format='three_tensor', returns='logits_only',
        description='FlexiModal MoE with top-k gating and modality bank'),

    # ═══ TEMPORAL DEEP MODELS (Phase 4/5) ═══
    ModelEntry('temporal_gru', TemporalGRU, group='temporal',
        init_kwargs={'input_dim': 69, 'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.3},
        input_format='fused', returns='logits_only',
        description='GRU temporal classifier'),
    ModelEntry('temporal_lstm', TemporalLSTM, group='temporal',
        init_kwargs={'input_dim': 69, 'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.3},
        input_format='fused', returns='logits_only',
        description='LSTM temporal classifier'),
    ModelEntry('cnn_lstm', CNNLSTMModel, group='temporal',
        init_kwargs={'input_dim': 69, 'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3},
        input_format='fused', returns='logits_only',
        description='CNN + LSTM hybrid'),
    ModelEntry('temporal_tcn', TemporalTCN, group='temporal',
        init_kwargs={'input_dim': 69, 'hidden_dim': 64, 'dropout': 0.3},
        input_format='fused', returns='logits_only',
        description='Temporal Convolutional Network (dilated)'),
    ModelEntry('temporal_transformer', TemporalTransformer, group='temporal',
        init_kwargs={'input_dim': 69, 'hidden_dim': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.3},
        input_format='fused', returns='logits_only',
        description='Transformer temporal classifier'),

    # ═══ EXPERT PIPELINE (Phase 6) ═══
    ModelEntry('expert_pipeline', ExpertPipelineModel, group='expert',
        init_kwargs={'subpart_dims': [9, 6, 18, 8, 13, 2, 2, 11], 'hidden_dim': 16},
        input_format='subparts', returns='logits_only',
        description='8 sub-part experts with gating router'),
]

def get_model_names(group=None):
    """Return sorted list of model names, optionally filtered by group."""
    names = [e.name for e in REGISTRY]
    if group:
        names = [e.name for e in REGISTRY if e.group == group]
    return sorted(names)

# ====================================================================
# INPUT ADAPTERS
# ====================================================================

def adapt_nine_tensor(batch, device):
    """Unpack 9-tensor batch to model inputs + labels."""
    eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, ds_id, w = \
        _unpack_batch(batch, device)
    return (eye, mouth, gface, sp, mfcc, qual, card, eda, soma,
            label, subj_id, ds_id, w)

def adapt_three_tensor(batch, device):
    """Combine 9 tensors into 3 modality tensors (face, voice, physio)."""
    eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, ds_id, w = \
        _unpack_batch(batch, device)
    face   = torch.cat([eye, mouth, gface], dim=-1)
    voice  = torch.cat([sp, mfcc, qual], dim=-1)
    physio = torch.cat([card, eda, soma], dim=-1)
    return (face, voice, physio, label, subj_id, ds_id, w)

def adapt_fused(batch, device):
    """Combine all 9 tensors into a single fused tensor."""
    eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, ds_id, w = \
        _unpack_batch(batch, device)
    fused = torch.cat([eye, mouth, gface, sp, mfcc, qual, card, eda, soma], dim=-1)
    return (fused, label, subj_id, ds_id, w)

def adapt_subparts(batch, device):
    """Split 9 tensors into 8 subparts (merge eda+soma)."""
    eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, ds_id, w = \
        _unpack_batch(batch, device)
    subparts = [eye, mouth, gface, sp, mfcc, qual, card, torch.cat([eda, soma], dim=-1)]
    return (subparts, label, subj_id, ds_id, w)

INPUT_ADAPTERS = {
    'nine_tensor':  adapt_nine_tensor,
    'three_tensor': adapt_three_tensor,
    'fused':        adapt_fused,
    'subparts':     adapt_subparts,
}

# ====================================================================
# TRAINING PIPELINE
# ====================================================================

TRAIN_CFG = {
    'seed':              42,
    'seq_len':           5,
    'batch_size':        256,
    'ssl_epochs':        4,
    'ft_epochs':         8,
    'lr_ssl':            1e-3,
    'lr_ft':             5e-4,
    'weight_decay':      1e-4,
    'modality_dropout':  0.15,
    'noise_std':         0.02,
    'lambda_conf':       0.15,
    'lambda_subj':       0.10,
    'lambda_dataset':    0.10,
    'lambda_attn':       0.05,
    'lambda_ssl':        0.05,
    'grl_alpha_subj':    0.02,
    'grl_alpha_ds':      0.05,
    'n_folds':           15,
    'dataset_weight_empathicschool': 0.3,
    'dataset_weight_stressid': 1.0,
    'dataset_weight_wesad': 1.0,
}

def build_model(model_entry: ModelEntry, group_dims: dict, n_subjects: int, n_datasets: int, device):
    """Instantiate a model from registry entry, adapting dims from enriched data."""
    kw = dict(model_entry.init_kwargs)
    # Override num_subjects with actual count
    if 'num_subjects' in kw:
        kw['num_subjects'] = n_subjects
    if 'num_datasets' in kw:
        kw['num_datasets'] = n_datasets
    # Override feature dims from enriched data
    face_dim = sum(group_dims[k] for k in ['face_eye', 'face_mouth', 'face_global_face'])
    voice_dim = sum(group_dims[k] for k in ['voice_spectral_prosody', 'voice_mfcc', 'voice_quality'])
    physio_dim = sum(group_dims[k] for k in ['physio_cardio', 'physio_eda', 'physio_somatic'])
    total_dim = face_dim + voice_dim + physio_dim
    for param_name in ('face_dim', 'voice_dim', 'physio_dim', 'total_feat_dim', 'input_dim'):
        if param_name in kw:
            if param_name == 'face_dim':
                kw[param_name] = face_dim
            elif param_name == 'voice_dim':
                kw[param_name] = voice_dim
            elif param_name == 'physio_dim':
                kw[param_name] = physio_dim
            elif param_name == 'total_feat_dim':
                kw[param_name] = total_dim
            elif param_name == 'input_dim':
                kw[param_name] = total_dim
    # Subpart dims
    if 'subpart_dims' in kw:
        kw['subpart_dims'] = [
            group_dims['face_eye'], group_dims['face_mouth'],
            group_dims['face_global_face'], group_dims['voice_spectral_prosody'],
            group_dims['voice_mfcc'], group_dims['voice_quality'],
            group_dims['physio_cardio'],
            group_dims['physio_eda'] + group_dims['physio_somatic'],
        ]
    model = model_entry.cls(**kw).to(device)
    return model

def forward_model(model, model_entry, batch, device):
    """Forward pass handling different input/output formats."""
    adapter = INPUT_ADAPTERS[model_entry.input_format]
    adapted = adapter(batch, device)

    if model_entry.input_format == 'nine_tensor':
        eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, ds_id, w = adapted
        if model_entry.returns == 'logits_subj_confidence':
            logits, subj_logits, confidence = model(
                eye, mouth, gface, sp, mfcc, qual, card, eda, soma, return_confidence=True)
        elif model_entry.returns == 'logits_confidence':
            logits, confidence = model(
                eye, mouth, gface, sp, mfcc, qual, card, eda, soma, return_confidence=True)
            subj_logits = logits.new_zeros(logits.size(0), 2)
        else:
            logits = model(eye, mouth, gface, sp, mfcc, qual, card, eda, soma)
            confidence = torch.ones(logits.size(0), device=device)
            subj_logits = logits.new_zeros(logits.size(0), 2)

    elif model_entry.input_format == 'three_tensor':
        face, voice, physio, label, subj_id, ds_id, w = adapted
        logits = model(face, voice, physio)
        confidence = torch.ones(logits.size(0), device=device)
        subj_logits = logits.new_zeros(logits.size(0), 2)

    elif model_entry.input_format == 'fused':
        fused, label, subj_id, ds_id, w = adapted
        logits = model(fused)
        confidence = torch.ones(logits.size(0), device=device)
        subj_logits = logits.new_zeros(logits.size(0), 2)

    elif model_entry.input_format == 'subparts':
        subparts, label, subj_id, ds_id, w = adapted
        logits, _, _ = model(subparts)
        confidence = torch.ones(logits.size(0), device=device)
        subj_logits = logits.new_zeros(logits.size(0), 2)

    else:
        raise ValueError(f"Unknown input_format: {model_entry.input_format}")

    return logits, subj_logits, confidence, label.long(), subj_id.long()

def run_benchmark_on_dataset(dataset_name, model_entry: ModelEntry, config: dict, device,
                              exclude_dataset=None):
    """Train and evaluate a single model on a single dataset with LOSO CV."""
    meta_path = os.path.join(ENRICHED_DIR, dataset_name, 'metadata.parquet')
    if not os.path.exists(meta_path):
        print(f"    SKIP {dataset_name}: enriched data not found")
        return None, []

    meta = pd.read_parquet(meta_path)

    held_out_meta = None
    train_subjects = None
    held_subjects = None
    if exclude_dataset and 'dataset' in meta.columns:
        held_out_meta = meta[meta['dataset'] == exclude_dataset].copy()
        meta = meta[meta['dataset'] != exclude_dataset].copy()
        if len(held_out_meta) == 0:
            held_out_meta = None
        else:
            n_held = held_out_meta['subject_id'].nunique()
            train_subjects = set(meta['subject_id'].unique())
            held_subjects = set(held_out_meta['subject_id'].unique())
            print(f"      Excluded {exclude_dataset}: {n_held} subjects, {len(held_out_meta)} windows (held-out)")

    meta = meta.reset_index(drop=True)
    subjects = sorted(meta['subject_id'].unique())
    n_subjects = len(subjects)

    if n_subjects < 2:
        print(f"    SKIP {dataset_name}: only {n_subjects} subjects")
        return None, []

    # Build dataset once to get GROUP_DIMS
    full_ds = SSVBDataset(dataset_name, seq_len=config['seq_len'], augment=False,
                          subject_filter=train_subjects)
    group_dims = {k: v.shape[-1] for k, v in full_ds.features.items()}
    n_datasets = len(full_ds.datasets)

    # Single-class subject filter (can't be held-out test subjects)
    subj_label_set = meta.groupby('subject_id')['label'].unique()
    single_class_subjs = set(subj_label_set[subj_label_set.apply(lambda x: len(x) < 2)].index)
    multi_class_subjs = [s for s in subjects if s not in single_class_subjs]

    if len(multi_class_subjs) < 2:
        print(f"    SKIP {dataset_name}: only {len(multi_class_subjs)} multi-class subjects")
        return None, []

    print(f"      Subjects: {n_subjects} total, {len(multi_class_subjs)} multi-class, "
          f"{len(single_class_subjs)} single-class (train-only)")

    # Stratified fold selection from multi-class pool
    rng = np.random.RandomState(config['seed'])
    if 'dataset' in meta.columns:
        subj_per_ds = meta.groupby('dataset')['subject_id'].unique()
        n_folds = min(config['n_folds'], len(multi_class_subjs))
        folds_per_ds = {}
        for ds_name in subj_per_ds.keys():
            eligible = [s for s in subj_per_ds[ds_name] if s in multi_class_subjs]
            folds_per_ds[ds_name] = max(1, int(round(n_folds * len(eligible) / len(multi_class_subjs))))
        diff = n_folds - sum(folds_per_ds.values())
        if diff > 0:
            largest = max(folds_per_ds, key=folds_per_ds.get)
            folds_per_ds[largest] += diff
        selected = []
        for ds_name, n_sel in folds_per_ds.items():
            eligible = [s for s in subj_per_ds[ds_name] if s in multi_class_subjs]
            rng.shuffle(eligible)
            selected.extend(eligible[:n_sel])
        selected = list(dict.fromkeys(selected))[:n_folds]
    else:
        n_folds = min(config['n_folds'], len(multi_class_subjs))
        rng.shuffle(multi_class_subjs)
        selected = multi_class_subjs[:n_folds]

    subj_to_idx = {s: i for i, s in enumerate(sorted(meta['subject_id'].unique()))}
    idx_to_subj = {v: k for k, v in subj_to_idx.items()}

    class IndexedDataset(Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    # Per-dataset weight map
    ds_weight_map = {}
    for ds_name in meta['dataset'].unique() if 'dataset' in meta.columns else [dataset_name]:
        key = f'dataset_weight_{ds_name}'
        ds_weight_map[ds_name] = config.get(key, 1.0)

    all_true, all_prob, all_conf, all_subj = [], [], [], []
    fold_metrics_list = []
    best_avg_auc = 0.0
    successful_folds = 0

    for fold, test_subj in enumerate(selected, 1):
        test_labels = meta[meta['subject_id'] == test_subj]['label'].values
        if len(np.unique(test_labels)) < 2:
            print(f"      SKIP fold {fold}/{len(selected)}: test subject {test_subj} has 1 class")
            continue

        print(f"\n      Fold {fold}/{len(selected)} (test: {test_subj})")

        train_subjs = [s for s in subjects if s != test_subj]
        train_idx = meta[meta['subject_id'].isin(train_subjs)].index.values
        test_idx  = meta[meta['subject_id'] == test_subj].index.values

        full_ds = SSVBDataset(dataset_name, seq_len=config['seq_len'], augment=False,
                              dataset_weights=ds_weight_map, subject_filter=train_subjects)
        train_ds = IndexedDataset(
            SSVBDataset(dataset_name, seq_len=config['seq_len'], augment=True,
                        noise_std=config['noise_std'],
                        modality_dropout=config['modality_dropout'],
                        dataset_weights=ds_weight_map, subject_filter=train_subjects),
            train_idx)
        test_ds = IndexedDataset(full_ds, test_idx)

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

        model = build_model(model_entry, group_dims, n_subjects, n_datasets, device)
        criterion_subj = nn.CrossEntropyLoss()

        # SSL pretraining (only models with explicit expert encoders: SSVB, ConvMoE-MF)
        has_experts = hasattr(model, 'exp_eye') or hasattr(model, 'enc_face')
        if config['ssl_epochs'] > 0 and has_experts:
            print(f"        SSL pretraining ({config['ssl_epochs']} epochs)")
            opt_ssl = optim.AdamW(model.parameters(), lr=config['lr_ssl'],
                                  weight_decay=config['weight_decay'])
            for ep in range(config['ssl_epochs']):
                model.train()
                ssl_loss = 0.0
                for batch in train_loader:
                    eye, mouth, gface, sp, mfcc, qual, card, eda, soma, _, subj_id, _, _ = \
                        _unpack_batch(batch, device)
                    if hasattr(model, 'exp_eye'):
                        e_eye = model.exp_eye(eye)
                        e_mouth = model.exp_mouth(mouth)
                        e_gface = model.exp_global_face(gface)
                        face_lat = torch.cat([e_eye, e_mouth, e_gface], dim=1)
                        e_sp = model.exp_spectral_prosody(sp)
                        e_mfcc = model.exp_mfcc(mfcc)
                        e_qual = model.exp_quality(qual)
                        voice_lat = torch.cat([e_sp, e_mfcc, e_qual], dim=1)
                        e_card = model.exp_cardio(card)
                        e_eda = model.exp_eda(eda)
                        e_soma = model.exp_somatic(soma)
                        physio_lat = torch.cat([e_card, e_eda, e_soma], dim=1)
                    elif hasattr(model, 'enc_face'):
                        face = torch.cat([eye, mouth, gface], dim=-1)
                        voice = torch.cat([sp, mfcc, qual], dim=-1)
                        physio = torch.cat([card, eda, soma], dim=-1)
                        face_lat = model.enc_face(face)
                        voice_lat = model.enc_voice(voice)
                        physio_lat = model.enc_physio(physio)
                    else:
                        continue
                    loss = (contrastive_loss(face_lat, subj_id) +
                            contrastive_loss(voice_lat, subj_id) +
                            contrastive_loss(physio_lat, subj_id))
                    opt_ssl.zero_grad()
                    loss.backward()
                    opt_ssl.step()
                    ssl_loss += loss.item()
                if (ep + 1) % 2 == 0:
                    print(f"        SSL epoch {ep+1}: loss={ssl_loss/max(len(train_loader),1):.4f}")

        # Supervised fine-tuning
        print(f"        Fine-tuning ({config['ft_epochs']} epochs)")
        opt_ft = optim.AdamW(model.parameters(), lr=model_entry.learning_rate,
                             weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=config['ft_epochs'])
        best_fold_auc = 0.0

        for ep in range(config['ft_epochs']):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                logits, subj_logits, confidence, label, subj_id = \
                    forward_model(model, model_entry, batch, device)
                opt_ft.zero_grad()
                probs = torch.softmax(logits, dim=1)
                y_onehot = nn.functional.one_hot(label, num_classes=2).float()
                probs_adj = confidence.unsqueeze(-1) * probs + (1 - confidence.unsqueeze(-1)) * y_onehot
                loss_stress = -torch.sum(y_onehot * torch.log(probs_adj + 1e-8), dim=1).mean()
                loss_conf = -torch.log(confidence + 1e-8).mean()
                loss = loss_stress + config['lambda_conf'] * loss_conf
                if model_entry.returns == 'logits_subj_confidence':
                    loss += config['lambda_subj'] * criterion_subj(subj_logits, subj_id)
                loss.backward()
                opt_ft.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            val_probs, val_true, val_conf = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    logits, _, confidence, label, _ = forward_model(model, model_entry, batch, device)
                    val_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    val_true.append(label.cpu().numpy())
                    val_conf.append(confidence.squeeze().cpu().numpy())
            val_probs = np.hstack(val_probs)
            val_true = np.hstack(val_true)
            val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.5
            scheduler.step()

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc

            if (ep + 1) % 4 == 0:
                print(f"        Epoch {ep+1}/{config['ft_epochs']}: loss={total_loss/max(len(train_loader),1):.4f}  val_AUC={val_auc:.4f}")

        # Final fold evaluation
        model.eval()
        fold_prob, fold_true, fold_conf = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, _, confidence, label, _ = forward_model(model, model_entry, batch, device)
                fold_prob.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                fold_true.append(label.cpu().numpy())
                fold_conf.append(confidence.squeeze().cpu().numpy())
        fold_prob = np.hstack(fold_prob)
        fold_true = np.hstack(fold_true)
        fold_conf = np.hstack(fold_conf)

        m = calculate_metrics(fold_true, fold_prob)
        m['mean_confidence'] = float(fold_conf.mean())
        fold_metrics_list.append(m)
        all_true.append(fold_true)
        all_prob.append(fold_prob)
        all_conf.append(fold_conf)
        all_subj.append(np.array([subj_to_idx[test_subj]] * len(fold_true)))
        successful_folds += 1
        print(f"      -> Fold {fold}: ACC={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    if successful_folds == 0:
        return None, []

    all_true = np.hstack(all_true)
    all_prob = np.hstack(all_prob)
    all_conf = np.hstack(all_conf)
    all_subj_h = np.hstack(all_subj)
    all_subj_str = np.array([idx_to_subj.get(int(s), str(int(s))) for s in all_subj_h])

    agg = calculate_metrics(all_true, all_prob)
    agg['mean_confidence'] = float(all_conf.mean())
    agg['n_folds'] = successful_folds
    print(f"\n      {dataset_name}: ACC={agg['accuracy']:.4f}  F1={agg['f1']:.4f}  AUC={agg['roc_auc']:.4f}")

    return agg, fold_metrics_list

# ====================================================================
# MAIN
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unified Research Model Benchmark')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model names')
    parser.add_argument('--skip', type=str, default=None, help='Comma-separated model groups to skip')
    parser.add_argument('--group', type=str, default=None, help='Only run models in this group')
    parser.add_argument('--list', action='store_true', help='List available models and exit')
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without training')
    parser.add_argument('--exclude-dataset', type=str, default=None,
                        help='Exclude dataset from combined training (e.g., empathicschool)')
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Model':30s} {'Group':12s} {'Input':14s} {'Returns':22s}  Description")
        print('-' * 100)
        for e in REGISTRY:
            print(f"{e.name:30s} {e.group:12s} {e.input_format:14s} {e.returns:22s}  {e.description}")
        print(f"\nGroups: {sorted(set(e.group for e in REGISTRY))}")
        return

    # Filter models
    if args.models:
        selected_names = set(args.models.split(','))
        models_to_run = [e for e in REGISTRY if e.name in selected_names]
        missing = selected_names - set(e.name for e in models_to_run)
        if missing:
            print(f"Warning: unknown models: {missing}")
    elif args.group:
        models_to_run = [e for e in REGISTRY if e.group == args.group]
    else:
        models_to_run = list(REGISTRY)

    if args.skip:
        skip_groups = set(args.skip.split(','))
        models_to_run = [e for e in models_to_run if e.group not in skip_groups]

    print(f"\n{'='*60}")
    print(f"  UNIFIED RESEARCH MODEL BENCHMARK")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Models: {len(models_to_run)}")
    for e in models_to_run:
        print(f"    - {e.name:30s} ({e.group})  {e.description}")
    print(f"{'='*60}")

    # Validate enriched data
    available_datasets = []
    for ds_name in ['stressid', 'wesad', 'combined']:
        meta_path = os.path.join(ENRICHED_DIR, ds_name, 'metadata.parquet')
        if os.path.exists(meta_path):
            meta = pd.read_parquet(meta_path)
            print(f"  {ds_name}: {len(meta)} windows, {meta['subject_id'].nunique()} subjects")
            available_datasets.append(ds_name)
        else:
            print(f"  {ds_name}: not found — skip")

    if len(available_datasets) < 1:
        print("ERROR: No enriched datasets found. Run build_enriched_training_data.py first.")
        sys.exit(1)

    if args.dry_run:
        print("\n  Dry-run: creating dataset and model...")
        ds = SSVBDataset('stressid', seq_len=5, augment=False)
        sample = ds[0]
        print(f"  Sample: {len(sample)} items (9 feats + label + subj_id + dataset_id + weight)")
        print(f"  Group dims: { {k: v.shape[-1] for k, v in ds.features.items()} }")
        model = build_model(models_to_run[0], {k: v.shape[-1] for k, v in ds.features.items()},
                            len(ds.subjects), len(ds.datasets), DEVICE)
        print(f"  Model {models_to_run[0].name}: {sum(p.numel() for p in model.parameters()):,} params")
        print("  Dry-run OK")
        return

    # Run benchmark
    all_summaries = {}
    exclude_ds = args.exclude_dataset

    for model_entry in models_to_run:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_entry.name} ({model_entry.group})")
        print(f"  {model_entry.description}")
        print(f"{'='*60}")

        model_results = {}
        per_dataset = {}

        for ds_name in available_datasets + (['combined'] if 'combined' not in available_datasets else []):
            if ds_name == 'combined' and 'combined' not in available_datasets:
                combined_path = os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')
                if not os.path.exists(combined_path):
                    continue

            print(f"\n  --- {ds_name.upper()} ---")
            exclude_this = exclude_ds if ds_name == 'combined' else None
            agg, fold_metrics = run_benchmark_on_dataset(
                ds_name, model_entry, TRAIN_CFG, DEVICE,
                exclude_dataset=exclude_this)

            if agg is None:
                continue

            model_results[ds_name] = agg
            per_dataset[ds_name] = {'aggregate': agg, 'folds': fold_metrics}

        if not model_results:
            print(f"  No results for {model_entry.name}")
            continue

        all_summaries[model_entry.name] = model_results

        # Save per-model results
        save_dir = os.path.join(REPORTS_DIR, model_entry.name)
        os.makedirs(save_dir, exist_ok=True)
        summary = {
            'model': model_entry.name,
            'group': model_entry.group,
            'description': model_entry.description,
            'params': model_entry.init_kwargs,
            'per_dataset': {k: v['aggregate'] for k, v in per_dataset.items()},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(save_dir, 'benchmark.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*60}")
    for model_name, results in all_summaries.items():
        parts = []
        for ds_name, m in results.items():
            parts.append(f"{ds_name}={m['roc_auc']:.4f}")
        print(f"  {model_name:30s}  {' | '.join(parts)}")

    # Build leaderboard
    leaderboard_rows = []
    for model_name, results in all_summaries.items():
        entry = next(e for e in REGISTRY if e.name == model_name)
        row = {'model': model_name, 'group': entry.group}
        for ds_name, m in results.items():
            row[f'{ds_name}_auc'] = m['roc_auc']
            row[f'{ds_name}_acc'] = m['accuracy']
            row[f'{ds_name}_f1'] = m['f1']
        leaderboard_rows.append(row)

    df_lb = pd.DataFrame(leaderboard_rows)
    if 'combined_auc' in df_lb.columns:
        df_lb = df_lb.sort_values('combined_auc', ascending=False)
    lb_path = os.path.join(REPORTS_DIR, 'leaderboard.csv')
    df_lb.to_csv(lb_path, index=False)
    print(f"\n  Leaderboard: {lb_path}")
    print(f"\n  Top-5 by combined AUC:")
    if 'combined_auc' in df_lb.columns:
        for _, row in df_lb.head(5).iterrows():
            print(f"    {row['model']:30s}  AUC={row['combined_auc']:.4f}  ACC={row['combined_acc']:.4f}  F1={row['combined_f1']:.4f}")

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS: {REPORTS_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
