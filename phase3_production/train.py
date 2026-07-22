"""
SSVB-CASA-AIS Production Training Pipeline

Trains the full 6-stage architecture on StressID, WESAD, and EmpathicSchool
datasets with proper LOSO cross-validation, data augmentation, and model
checkpointing for deployment.

Usage:
    python train_ssvb_production.py                    # full pipeline
    python train_ssvb_production.py --dataset stressid  # single dataset
    python train_ssvb_production.py --dry-run           # validate setup only
"""
import os, sys, time, json, warnings, copy, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, average_precision_score,
                             precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -- Paths ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from models.ssvb_casa_ais import SSVBCASA_AIS
from models.conv_moe_mf import ConvMoE_MF


# ===========================================================================
# CNN Baseline (plain 1D-CNN for ablation comparison)
# ===========================================================================
class CNNBaseline(nn.Module):
    """Plain 1D-CNN baseline. No GRU, no attention, no gating, no GRL.
    Concatenates all 9 sub-modality features into a single stream.

    Used in ablation: measures accuracy gap between simple CNN and
    ConvMoE-MF under identical LOSO protocol. Higher gap = stronger
    evidence for ConvMoE-MF's identity-suppression value.
    """
    def __init__(self, total_feat_dim=69, hidden_dims=[64, 32, 16], num_classes=2, num_subjects=91):
        super().__init__()
        layers = []
        in_ch = total_feat_dim
        for h in hidden_dims:
            layers.extend([
                nn.Conv1d(in_ch, h, kernel_size=3, padding=1),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ])
            in_ch = h
        self.conv_stack = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.num_subjects = num_subjects

    def _forward_cat(self, *feat_tensors):
        cat = torch.cat(feat_tensors, dim=-1)  # [B, T, total_feat_dim]
        x = cat.permute(0, 2, 1)              # [B, total_feat_dim, T]
        x = self.conv_stack(x)                # [B, 16, T]
        x = x.mean(dim=2)                     # GAP: [B, 16]
        return self.classifier(x), x

    def forward(self, eye, mouth, global_face,
                spectral_prosody, mfcc, quality,
                cardio, eda, somatic,
                return_all=False, return_confidence=False):
        logits, _ = self._forward_cat(
            eye, mouth, global_face,
            spectral_prosody, mfcc, quality,
            cardio, eda, somatic)
        B = logits.size(0)
        dummy_conf = torch.ones(B, device=logits.device)
        subj_logits = logits.new_zeros(B, self.num_subjects)
        if return_confidence:
            return logits, subj_logits, dummy_conf
        if return_all:
            return {
                "stress_logits":   logits,
                "confidence":      dummy_conf,
                "subj_logits":     subj_logits,
                "dataset_logits":  logits.new_zeros(B, 3),
                "gate_weights":    logits.new_ones(B, 1),
                "fused_embedding": logits.new_zeros(B, 16),
            }
        return logits, dummy_conf

    def forward_with_latent(self, eye=None, mouth=None, global_face=None,
                            spectral_prosody=None, mfcc=None, quality=None,
                            cardio=None, eda=None, somatic=None):
        logits, latent = self._forward_cat(
            eye, mouth, global_face,
            spectral_prosody, mfcc, quality,
            cardio, eda, somatic)
        return logits, latent


class CNNBaselineGRL(nn.Module):
    """CNNBaseline + subject-GRL. No MoE, no confidence head, no dataset-GRL.
    Isolates the effect of GRL-driven identity suppression on a simple CNN.
    """
    def __init__(self, total_feat_dim=69, hidden_dims=[64, 32, 16],
                 num_classes=2, num_subjects=91, grl_alpha=0.02):
        super().__init__()
        layers = []
        in_ch = total_feat_dim
        for h in hidden_dims:
            layers.extend([nn.Conv1d(in_ch, h, kernel_size=3, padding=1),
                           nn.BatchNorm1d(h), nn.ReLU()])
            in_ch = h
        self.conv_stack = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.subj_head = nn.Linear(hidden_dims[-1], num_subjects)
        self.grl_alpha = grl_alpha
        self.num_subjects = num_subjects

    def _forward_cat(self, *feat_tensors):
        cat = torch.cat(feat_tensors, dim=-1)
        x = cat.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = x.mean(dim=2)
        return self.classifier(x), x

    def forward(self, eye, mouth, global_face,
                spectral_prosody, mfcc, quality,
                cardio, eda, somatic,
                return_all=False, return_confidence=False):
        logits, latent = self._forward_cat(
            eye, mouth, global_face,
            spectral_prosody, mfcc, quality,
            cardio, eda, somatic)
        from models.conv_moe_mf import grad_reverse
        rev = grad_reverse(latent, self.grl_alpha)
        subj_logits = self.subj_head(rev)
        B = logits.size(0)
        dummy_conf = torch.ones(B, device=logits.device)
        if return_confidence:
            return logits, subj_logits, dummy_conf
        if return_all:
            return {"stress_logits": logits, "confidence": dummy_conf,
                    "subj_logits": subj_logits,
                    "dataset_logits": logits.new_zeros(B, 3),
                    "gate_weights": logits.new_ones(B, 1),
                    "fused_embedding": latent}
        return logits, dummy_conf


CERTIFIED_DIR  = os.path.join(PROJECT_ROOT, 'data', 'processed', 'certified_data')
PIPELINE_DATA   = os.path.join(PROJECT_ROOT, 'research', 'pipeline', 'data')
REPORTS_DIR     = os.path.join(os.path.dirname(__file__), 'results')
CHECKPOINT_DIR  = os.path.join(REPORTS_DIR, 'checkpoints')
DEPLOY_DIR      = os.path.join(REPORTS_DIR, 'deploy')
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEPLOY_DIR, exist_ok=True)

# -- Config -----------------------------------------------------------------
CONFIG = {
    'seed':               42,
    'seq_len':            5,
    'batch_size':         256,
    'ssl_epochs':         4,
    'ft_epochs':          8,
    'lr_ssl':             1e-3,
    'lr_ft':              5e-4,
    'weight_decay':       1e-4,
    'hidden_dim':         16,
    'modality_dropout':   0.15,
    'noise_std':          0.02,
    'lambda_conf':        0.15,
    'lambda_subj':        0.10,
    'lambda_dataset':     0.10,
    'lambda_attn':        0.05,
    'lambda_ssl':         0.05,
    'grl_alpha_subj':     0.02,
    'grl_alpha_ds':       0.05,
    'dataset_weight_empathicschool': 0.3,
    'dataset_weight_stressid': 1.0,
    'dataset_weight_wesad': 1.0,
    'n_folds':            15,
    'model_type':         'conv_moe_mf',  # 'ssvb', 'conv_moe_mf', 'cnn_baseline'
    'device':             'cuda' if torch.cuda.is_available() else 'cpu',
}

# Enriched training data directory (from build_enriched_training_data.py)
ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')

# 10 sub-modality groups (69 features after 3 privacy exclusions)
# Keys match the channel group names from build_enriched_training_data.py
SUB_GROUPS = {
    'face':  ['face_eye', 'face_mouth', 'face_global_face'],
    'voice': ['voice_spectral_prosody', 'voice_mfcc', 'voice_quality'],
    'physio':['physio_cardio', 'physio_eda', 'physio_somatic'],
}
# Input dimensions for each group (auto-detected from data)
GROUP_DIMS = None

DATASET_CONFIG = {
    'stressid': {
        'type': 'certified',
        'path': CERTIFIED_DIR,
        'modalities': ['face', 'voice', 'physio'],
    },
    'wesad': {
        'type': 'pipeline',
        'path': os.path.join(PIPELINE_DATA, 'wesad'),
        'modalities': ['physio'],
    },
    'empathicschool': {
        'type': 'pipeline',
        'path': os.path.join(PIPELINE_DATA, 'empathicschool'),
        'modalities': ['face', 'physio'],
    },
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])

print(f"Device: {CONFIG['device']}")


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_certified_dataset(data_dir):
    """Load certified CSVs and merge into a single DataFrame with
    calibration-normalised 8 sub-modality features."""
    # Load per-modality CSVs
    dfs = {}
    for mod in ['face', 'voice', 'physio']:
        path = os.path.join(data_dir, f"{mod}_certified.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.drop(columns=[c for c in ['video_id', 'window_start', 'window_end']
                             if c in df.columns], inplace=True, errors='ignore')
            for col in ['subject_id', 'task_id']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower().str.strip()
            if 'window_index' in df.columns:
                df['window_index'] = df['window_index'].astype(int)
            dfs[mod] = df

    if not dfs:
        raise FileNotFoundError(f"No certified CSVs found in {data_dir}")

    common_keys = ['subject_id', 'task_id', 'window_index', 'label']
    df_merged = None
    for mod, df in dfs.items():
        if df_merged is None:
            df_merged = df
        else:
            df_merged = df_merged.merge(df, on=[k for k in common_keys if k in df.columns],
                                        how='outer', suffixes=('', f'_{mod}'))

    df_merged.sort_values(['subject_id', 'task_id', 'window_index'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    df_merged.fillna(0, inplace=True)

    # Ensure all expected sub-modality columns exist
    for mod_subs in SUB_GROUPS.values():
        for group_feats in mod_subs.values():
            for f in group_feats:
                if f not in df_merged.columns:
                    df_merged[f] = 0.0

    # Feature columns (everything except metadata)
    excluded = ['video_id', 'window_start', 'window_end']
    meta_cols = ['subject_id', 'task_id', 'window_index', 'label', 'dataset']
    feat_cols = [c for c in df_merged.columns if c not in meta_cols and c not in excluded]

    # Calibration normalisation: subtract each subject's calm mean
    for subj in df_merged['subject_id'].unique():
        mask = df_merged['subject_id'] == subj
        calm = df_merged.loc[mask & (df_merged['label'] == 0)]
        if len(calm) > 0:
            means = calm[feat_cols].mean()
        else:
            means = df_merged.loc[mask, feat_cols].mean()
        df_merged.loc[mask, feat_cols] = df_merged.loc[mask, feat_cols] - means

    return df_merged


def load_pipeline_dataset(data_dir):
    """Load pre-extracted parquet data from pipeline data directory.

    The pipeline stores aggregated flat features (mean/std/min/max/range)
    which don't map directly to our per-frame sub-modality contract.
    Returns None — only certified CSVs support the 8 sub-modality split.
    """
    print(f"    NOTE: Pipeline data format not compatible with sub-modality split."
          f" Convert to certified CSV format first. Skipping.")
    return None


def load_dataset(name):
    """Load a single dataset by name and return a DataFrame with standard columns."""
    cfg = DATASET_CONFIG[name]
    if cfg['type'] == 'certified':
        df = load_certified_dataset(cfg['path'])
    else:
        df = load_pipeline_dataset(cfg['path'])
    if df is None or len(df) == 0:
        return None
    df['dataset'] = name
    return df


def get_enriched_subject_ids(dataset_name):
    """Return unique subject IDs from enriched metadata."""
    path = os.path.join(ENRICHED_DIR, dataset_name, "metadata.parquet")
    if not os.path.exists(path):
        return []
    meta = pd.read_parquet(path)
    return sorted(meta['subject_id'].unique())


def get_enriched_split(dataset_name, test_subject):
    """Return train/test indices for a given test subject."""
    path = os.path.join(ENRICHED_DIR, dataset_name, "metadata.parquet")
    meta = pd.read_parquet(path)
    test_idx = meta[meta['subject_id'] == test_subject].index.values
    train_idx = meta[meta['subject_id'] != test_subject].index.values
    return train_idx, test_idx


def load_all_datasets(datasets=None):
    """Validate that enriched data exists for requested datasets."""
    if datasets is None:
        datasets = [d for d in ['stressid', 'wesad', 'empathicschool', 'combined']
                    if os.path.isdir(os.path.join(ENRICHED_DIR, d))]
    for ds_name in datasets:
        path = os.path.join(ENRICHED_DIR, ds_name, "metadata.parquet")
        if os.path.exists(path):
            meta = pd.read_parquet(path)
            print(f"  {ds_name}: {len(meta)} windows, {meta['subject_id'].nunique()} subjects")
        else:
            print(f"  {ds_name}: not found — run build_enriched_training_data.py first")
            raise FileNotFoundError(f"Missing enriched data: {path}")
    return datasets


# ===========================================================================
# DATASET (sequence builder with augmentation)
# ===========================================================================

class SSVBDataset(Dataset):
    """Reads enriched pipeline sequences [N, 30, 72] mapped to 10 sub-modality
    groups.  Provides per-window [T, feat_dim] tensors for training."""

    def __init__(self, dataset_name, seq_len=30, augment=False, noise_std=0.02,
                 modality_dropout=0.15, dataset_weights=None):
        self.seq_len = seq_len
        self.augment = augment
        self.noise_std = noise_std
        self.modality_dropout = modality_dropout
        self.dataset_name = dataset_name
        self.dataset_weights = dataset_weights or {}

        data_dir = os.path.join(ENRICHED_DIR, dataset_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Enriched data not found: {data_dir}. "
                                    f"Run build_enriched_training_data.py first.")

        # Load feature arrays
        loaded = np.load(os.path.join(data_dir, "sequences.npz"))
        self.group_keys = sorted(loaded.keys())
        self.features = {k: loaded[k].astype(np.float32) for k in self.group_keys}

        # Load metadata
        self.meta = pd.read_parquet(os.path.join(data_dir, "metadata.parquet"))
        self.subjects = sorted(self.meta['subject_id'].unique())
        self.subj_to_idx = {s: i for i, s in enumerate(self.subjects)}

        # Dataset mapping (for dataset-invariant adversarial head)
        if 'dataset' in self.meta.columns:
            self.datasets = sorted(self.meta['dataset'].unique())
            self.ds_to_idx = {d: i for i, d in enumerate(self.datasets)}
        else:
            self.datasets = [dataset_name]
            self.ds_to_idx = {dataset_name: 0}

        # Store group dimensions
        global GROUP_DIMS
        GROUP_DIMS = {k: v.shape[-1] for k, v in self.features.items()}

        N = len(self.meta)
        self.labels = self.meta['label'].values.astype(np.int64)
        self.subj_ids = self.meta['subject_id'].values
        self.window_indices = self.meta['window_index'].values

    def __len__(self):
        return len(self.meta)

    def _augment(self, tensors):
        if self.noise_std > 0:
            tensors = [t + torch.randn_like(t) * self.noise_std for t in tensors]
        return tensors

    def __getitem__(self, idx):
        # Each feature is [T=30, feat_dim]
        tensors = [torch.FloatTensor(np.nan_to_num(self.features[k][idx], nan=0.0))
                   for k in self.group_keys]

        # Modality dropout at the modality level (face/voice/physio)
        if self.augment and self.modality_dropout > 0:
            r = np.random.rand()
            if r < self.modality_dropout / 3:
                # Drop face (groups 0-2)
                for i in range(3):
                    tensors[i] = tensors[i] * 0
            elif r < 2 * self.modality_dropout / 3:
                # Drop voice (groups 3-5)
                for i in range(3, 6):
                    tensors[i] = tensors[i] * 0
            elif r < self.modality_dropout:
                # Drop physio (groups 6-8)
                for i in range(6, 9):
                    tensors[i] = tensors[i] * 0

        if self.augment:
            tensors = self._augment(tensors)

        label = self.labels[idx]
        subj_id = self.subj_to_idx.get(self.subj_ids[idx], 0)
        dataset_name = self.meta['dataset'].iloc[idx]
        dataset_id = self.ds_to_idx.get(dataset_name, 0)
        weight = self.dataset_weights.get(dataset_name, 1.0)
        return (*tensors, label, subj_id, dataset_id, weight)


# ===========================================================================
# LOSSES
# ===========================================================================

def contrastive_loss(embeddings, subject_ids, temperature=0.1):
    """InfoNCE contrastive loss: positive pairs = same subject."""
    embeddings = embeddings / (torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    subj = subject_ids.view(-1, 1)
    mask = torch.eq(subj, subj.T).float()
    diag = torch.eye(mask.shape[0], device=embeddings.device)
    mask = mask - diag
    exp_sim = torch.exp(sim_matrix)
    sum_exp = torch.sum(exp_sim, dim=1, keepdim=True) - torch.diag(exp_sim).view(-1, 1)
    log_prob = sim_matrix - torch.log(sum_exp + 1e-8)
    pos_cnt = torch.sum(mask, dim=1)
    pos_cnt_safe = torch.where(pos_cnt > 0, pos_cnt, torch.ones_like(pos_cnt))
    loss = -torch.sum(log_prob * mask, dim=1) / pos_cnt_safe
    loss = torch.where(pos_cnt > 0, loss, torch.zeros_like(loss))
    return torch.mean(loss)


def attention_alignment_loss(cross_outputs):
    """Maximise cosine similarity between cross-attention outputs."""
    f = cross_outputs['f_re']
    v = cross_outputs['v_re']
    p = cross_outputs['p_re']
    fn = f / (torch.norm(f, dim=-1, keepdim=True) + 1e-8)
    vn = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
    pn = p / (torch.norm(p, dim=-1, keepdim=True) + 1e-8)
    sim_fv = (fn * vn).sum(dim=-1).mean()
    sim_fp = (fn * pn).sum(dim=-1).mean()
    sim_vp = (vn * pn).sum(dim=-1).mean()
    return 3.0 - (sim_fv + sim_fp + sim_vp)


# ===========================================================================
# METRICS & REPORTING
# ===========================================================================

def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'avg_precision': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'mse':       ((y_true - y_prob) ** 2).mean(),
        'mae':       np.abs(y_true - y_prob).mean(),
    }


def find_optimal_threshold(y_true, y_prob, metric='f1', thresholds=None):
    """Sweep thresholds and return the best one for a given metric."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_score = 0.0
    results = []
    for t in thresholds:
        m = calculate_metrics(y_true, y_prob, threshold=t)
        score = m.get(metric, m['f1'])
        results.append({'threshold': float(t), 'precision': m['precision'],
                        'recall': m['recall'], 'f1': m['f1']})
        if score > best_score:
            best_score = score
            best_thresh = float(t)
    return best_thresh, results


def per_subject_metrics(df_results):
    """Compute accuracy per subject, report mean and std."""
    subj_accs = df_results.groupby('subject_id').apply(
        lambda g: accuracy_score(g['true'], g['pred']))
    return {
        'subject_acc_mean': float(subj_accs.mean()),
        'subject_acc_std':  float(subj_accs.std()),
        'subject_acc_min':  float(subj_accs.min()),
        'subject_acc_max':  float(subj_accs.max()),
    }


def per_dataset_metrics(df_results):
    """Compute accuracy per dataset."""
    ds_accs = df_results.groupby('dataset').apply(
        lambda g: accuracy_score(g['true'], g['pred']))
    return {str(k): float(v) for k, v in ds_accs.items()}


def generate_plots(y_true, y_prob, model_name, save_dir, threshold=0.5):
    """ROC curve + PR-ROC + confusion matrix + classification report."""
    y_pred = (y_prob >= threshold).astype(int)

    # ROC curve
    plt.figure(figsize=(6, 5), dpi=150)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='#1b7a60', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='#999', lw=1.5)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_auc.png'), bbox_inches='tight')
    plt.close()

    # PR-ROC curve
    plt.figure(figsize=(6, 5), dpi=150)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.plot(recall, precision, color='#1b7a60', lw=2, label=f'AP = {ap:.4f}')
    plt.axhline(y=y_true.mean(), color='#999', linestyle='--',
                label=f'Baseline (pos ratio={y_true.mean():.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'PR-ROC - {model_name}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_roc.png'), bbox_inches='tight')
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4), dpi=150)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f'Confusion - {model_name}')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.ylabel('True'); plt.xlabel('Pred')
    plt.xticks([0, 1], ['Calm', 'Stress'])
    plt.yticks([0, 1], ['Calm', 'Stress'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # Classification report (text)
    report = classification_report(y_true, y_pred, target_names=['Calm', 'Stress'],
                                   digits=4)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f'Classification Report — {model_name}\n')
        f.write(f'{"="*50}\n\n')
        f.write(report)
        f.write(f'\nAUC-ROC: {auc:.4f}\n')
        f.write(f'Average Precision (AP): {ap:.4f}\n')


# ===========================================================================
# TRAINING
# ===========================================================================

def _unpack_batch(batch, device):
    """Unpack a batch from SSVBDataset (9 feats + label + subj_id + dataset_id + weight)."""
    t = [x.to(device) for x in batch]
    label, subj_id, dataset_id, weight = t[-4], t[-3], t[-2], t[-1]
    feats = t[:-4]
    # group_keys sorted: face_eye(0), face_global_face(1), face_mouth(2),
    #   physio_cardio(3), physio_eda(4), physio_somatic(5),
    #   voice_mfcc(6), voice_quality(7), voice_spectral_prosody(8)
    # model expects: eye, mouth, gface, spectral_prosody, mfcc, quality, cardio, eda, soma
    return (
        feats[0],   # eye
        feats[2],   # mouth
        feats[1],   # global_face
        feats[8],   # spectral_prosody
        feats[6],   # mfcc
        feats[7],   # quality
        feats[3],   # cardio
        feats[4],   # eda
        feats[5],   # somatic
        label.long(), subj_id.long(), dataset_id.long(), weight,
    )


def train_ssl_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        eye, mouth, gface, sp, mfcc, qual, card, eda, soma, _, subj_id, _, _ = _unpack_batch(batch, device)
        optimizer.zero_grad()

        # Check model type: SSVB has per-sub-modality experts, ConvMoE-MF has modality encoders
        if hasattr(model, 'exp_eye'):
            # SSVB-CASA-AIS style: 9 per-sub-modality experts
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
        else:
            # ConvMoE-MF style: 3 modality encoders (combines sub-modalities internally)
            face   = torch.cat([eye, mouth, gface], dim=-1)
            voice  = torch.cat([sp, mfcc, qual], dim=-1)
            physio = torch.cat([card, eda, soma], dim=-1)
            face_lat   = model.enc_face(face)
            voice_lat  = model.enc_voice(voice)
            physio_lat = model.enc_physio(physio)

        loss = (contrastive_loss(face_lat, subj_id) +
                contrastive_loss(voice_lat, subj_id) +
                contrastive_loss(physio_lat, subj_id))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_supervised_epoch(model, loader, optimizer, criterion_subj, criterion_ds, device, config):
    model.train()
    total_loss = 0.0
    lambda_conf = config['lambda_conf']
    lambda_subj = config['lambda_subj']
    lambda_dataset = config.get('lambda_dataset', 0.0)

    for batch in loader:
        eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, dataset_id, weight = _unpack_batch(batch, device)
        optimizer.zero_grad()

        stress_logits, subj_logits, confidence = model(
            eye, mouth, gface, sp, mfcc, qual, card, eda, soma,
            return_confidence=True)

        probs = torch.softmax(stress_logits, dim=1)
        y_onehot = nn.functional.one_hot(label, num_classes=2).float()
        probs_adj = confidence.unsqueeze(-1) * probs + (1 - confidence.unsqueeze(-1)) * y_onehot
        loss_stress_per_sample = -torch.sum(y_onehot * torch.log(probs_adj + 1e-8), dim=1)
        loss_stress = (loss_stress_per_sample * weight).mean()
        loss_conf = -torch.log(confidence + 1e-8).mean()
        loss_sup = loss_stress + lambda_conf * loss_conf
        loss_subj = criterion_subj(subj_logits, subj_id)

        # Dataset-adversarial loss (only if model has dataset_head)
        loss_ds = torch.tensor(0.0, device=device)
        if hasattr(model, 'dataset_head') and lambda_dataset > 0:
            out_all = model(eye, mouth, gface, sp, mfcc, qual, card, eda, soma, return_all=True)
            loss_ds = criterion_ds(out_all['dataset_logits'], dataset_id)

        loss = loss_sup + lambda_subj * loss_subj + lambda_dataset * loss_ds

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_true, all_conf, all_subjs = [], [], [], []
    for batch in loader:
        eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, _, _ = _unpack_batch(batch, device)
        stress_logits, _, confidence = model(
            eye, mouth, gface, sp, mfcc, qual, card, eda, soma,
            return_confidence=True)
        probs = torch.softmax(stress_logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_true.append(label.cpu().numpy())
        all_conf.append(confidence.squeeze().cpu().numpy())
        all_subjs.append(subj_id.cpu().numpy())
    return (np.hstack(all_probs), np.hstack(all_true),
            np.hstack(all_conf), np.hstack(all_subjs))


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'metrics': metrics,
        'config': CONFIG,
    }, path)


def export_deployment_weights(model, path, optimal_threshold=0.5):
    """Export model weights in a format loadable by RuntimeEngine."""
    torch.save(model.state_dict(), path)
    arch_name = CONFIG.get('model_type', 'ssvb')
    arch_labels = {
        'conv_moe_mf': 'ConvMoE-MF',
        'ssvb': 'SSVB-CASA-AIS (Hybrid MoE)',
        'cnn_baseline': 'CNNBaseline',
        'cnn_baseline_grl': 'CNNBaseline+GRL',
    }
    meta = {
        'architecture': arch_labels.get(arch_name, 'Unknown'),
        'model_type': arch_name,
        'hidden_dim': CONFIG['hidden_dim'],
        'num_subjects': CONFIG.get('n_subjects', 65),
        'trained_on': str(CONFIG.get('trained_on', 'unknown')),
        'optimal_threshold': optimal_threshold,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
    }
    with open(path.replace('.pt', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ===========================================================================
# CROSS-VALIDATION
# ===========================================================================

def run_cross_validation(dataset_name, config):
    """Run stratified LOSO CV. Selects test subjects proportionally from
    each dataset to avoid degenerate folds from random sampling."""
    device = torch.device(config['device'])

    # Load enriched metadata to get subjects
    meta_path = os.path.join(ENRICHED_DIR, dataset_name, "metadata.parquet")
    meta = pd.read_parquet(meta_path)
    subjects = sorted(meta['subject_id'].unique())
    num_subjects = len(subjects)

    if len(subjects) < 2:
        print(f"  SKIP {dataset_name}: only {len(subjects)} subjects")
        return None, [], None, None

    # Stratified subject selection: pick proportionally from each dataset
    rng = np.random.RandomState(config['seed'])
    if 'dataset' in meta.columns:
        subj_per_ds = meta.groupby('dataset')['subject_id'].unique()
        n_folds = min(config['n_folds'], len(subjects))
        # Allocate folds per dataset proportionally
        total = len(meta)
        folds_per_ds = {}
        for ds, subjs in subj_per_ds.items():
            pct = len(subjs) / num_subjects
            folds_per_ds[ds] = max(1, int(round(n_folds * pct)))
        # Adjust total to match n_folds
        diff = n_folds - sum(folds_per_ds.values())
        if diff > 0:
            # Give extra folds to largest dataset
            largest = max(folds_per_ds, key=folds_per_ds.get)
            folds_per_ds[largest] += diff
        # Select subjects from each dataset
        selected = []
        for ds, n_sel in folds_per_ds.items():
            pool = list(subj_per_ds[ds])
            rng.shuffle(pool)
            selected.extend(pool[:n_sel])
        # Deduplicate (in case of overlap) and cap at n_folds
        selected = list(dict.fromkeys(selected))[:n_folds]
    else:
        n_folds = min(config['n_folds'], len(subjects))
        rng.shuffle(subjects)
        selected = subjects[:n_folds]

    all_true, all_prob, all_conf, all_subj = [], [], [], []
    fold_metrics = []
    best_avg_auc = 0.0
    best_state = None
    successful_folds = 0

    for fold, test_subj in enumerate(selected, 1):
        train_subjs = [s for s in subjects if s != test_subj]
        # Check that test set has both classes (otherwise AUC is meaningless)
        test_labels = meta[meta['subject_id'] == test_subj]['label'].values
        if len(np.unique(test_labels)) < 2:
            print(f"\n  SKIP fold {fold}/{len(selected)}: test subject {test_subj} has only 1 class")
            continue

        print(f"\n{'='*60}")
        print(f"  {dataset_name} — Fold {fold}/{len(selected)} (test: {test_subj})")
        print(f"{'='*60}")

        # LOSO split: one subject for test, rest for train
        train_idx = meta[meta['subject_id'].isin(train_subjs)].index.values
        test_idx  = meta[meta['subject_id'] == test_subj].index.values

        # Build datasets that index by row index
        class IndexedDataset(Dataset):
            def __init__(self, base_ds, indices):
                self.base = base_ds
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]

        # Build per-dataset weight map (downweight EmpathicSchool)
        ds_weight_map = {}
        for ds_name in meta['dataset'].unique() if 'dataset' in meta.columns else [dataset_name]:
            key = f'dataset_weight_{ds_name}'
            ds_weight_map[ds_name] = config.get(key, 1.0)

        # Reuse a single SSVBDataset for the whole dataset (handles features)
        full_ds = SSVBDataset(dataset_name, seq_len=config['seq_len'],
                              augment=False, dataset_weights=ds_weight_map)
        train_ds = IndexedDataset(SSVBDataset(dataset_name, seq_len=config['seq_len'],
                                  augment=True, noise_std=config['noise_std'],
                                  modality_dropout=config['modality_dropout'],
                                  dataset_weights=ds_weight_map),
                                  train_idx)
        test_ds  = IndexedDataset(full_ds, test_idx)

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_ds, batch_size=config['batch_size'],
                                  shuffle=False, num_workers=0)

        model_type = config.get('model_type', 'ssvb')
        if model_type == 'conv_moe_mf':
            num_datasets = len(meta['dataset'].unique()) if 'dataset' in meta.columns else 1
            model = ConvMoE_MF(hidden_dim=config['hidden_dim'], embed_dim=8,
                               num_subjects=num_subjects,
                               num_datasets=num_datasets,
                               grl_alpha_subj=config.get('grl_alpha_subj', 0.02),
                               grl_alpha_ds=config.get('grl_alpha_ds', 0.02)).to(device)
        elif model_type == 'ssvb':
            model = SSVBCASA_AIS(hidden_dim=config['hidden_dim'],
                                 num_subjects=num_subjects).to(device)
        elif model_type == 'cnn_baseline':
            total_feat_dim = sum(v for v in GROUP_DIMS.values()) if GROUP_DIMS else 69
            model = CNNBaseline(total_feat_dim=total_feat_dim, num_subjects=num_subjects).to(device)
        elif model_type == 'cnn_baseline_grl':
            total_feat_dim = sum(v for v in GROUP_DIMS.values()) if GROUP_DIMS else 69
            model = CNNBaselineGRL(total_feat_dim=total_feat_dim,
                                   num_subjects=num_subjects,
                                   grl_alpha=config.get('grl_alpha_subj', 0.02)).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        criterion_subj = nn.CrossEntropyLoss()
        criterion_ds = nn.CrossEntropyLoss()

        # -- Stage 1: SSL Pretraining (SSVB + ConvMoE-MF; CNN variants skip) --
        if config['ssl_epochs'] > 0 and model_type not in ('cnn_baseline', 'cnn_baseline_grl'):
            print(f"  Stage 1: SSL contrastive pretraining ({config['ssl_epochs']} epochs)")
            opt_ssl = optim.AdamW(model.parameters(), lr=config['lr_ssl'],
                                  weight_decay=config['weight_decay'])
            for ep in range(config['ssl_epochs']):
                loss = train_ssl_epoch(model, train_loader, opt_ssl, device)
                print(f"    Epoch {ep+1}/{config['ssl_epochs']}  SSL loss: {loss:.4f}")

        # -- Stage 2: Supervised Fine-Tuning --
        print(f"  Stage 2: Supervised fine-tuning ({config['ft_epochs']} epochs)")
        opt_ft = optim.AdamW(model.parameters(), lr=config['lr_ft'],
                             weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=config['ft_epochs'])
        best_fold_auc = 0.0
        best_fold_state = None

        for ep in range(config['ft_epochs']):
            train_loss = train_supervised_epoch(model, train_loader, opt_ft,
                                                 criterion_subj, criterion_ds, device, config)
            val_prob, val_true, val_conf, _ = evaluate(model, test_loader, device)
            val_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.5
            scheduler.step()

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_fold_state = copy.deepcopy(model.state_dict())

            if (ep + 1) % 4 == 0 or ep == 0:
                print(f"    Epoch {ep+1}/{config['ft_epochs']}  loss: {train_loss:.4f}  val AUC: {val_auc:.4f}")

        # Restore best fold state and evaluate
        if best_fold_state is not None:
            model.load_state_dict(best_fold_state)
        fold_prob, fold_true, fold_conf, fold_subj = evaluate(model, test_loader, device)

        # Ensure shapes match
        min_len = min(len(fold_true), len(fold_prob), len(fold_conf))
        fold_true = fold_true[:min_len]
        fold_prob = fold_prob[:min_len]
        fold_conf = fold_conf[:min_len]

        metrics = calculate_metrics(fold_true, fold_prob)
        metrics['mean_confidence'] = float(fold_conf.mean()) if len(fold_conf) > 0 else 0.0
        fold_metrics.append(metrics)

        all_true.append(fold_true)
        all_prob.append(fold_prob)
        all_conf.append(fold_conf)
        all_subj.append(fold_subj)

        successful_folds += 1
        print(f"  → Fold {fold}: ACC={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['roc_auc']:.4f}  Conf={metrics['mean_confidence']:.4f}")

        # Track best model across folds
        if metrics['roc_auc'] > best_avg_auc:
            best_avg_auc = metrics['roc_auc']
            best_state = copy.deepcopy(model.state_dict())

    # Aggregate
    if successful_folds == 0:
        print(f"\n  ERROR: No valid folds completed")
        return None, [], None, None

    all_true = np.hstack(all_true)
    all_prob = np.hstack(all_prob)
    all_conf = np.hstack(all_conf)

    agg = calculate_metrics(all_true, all_prob)
    agg['mean_confidence'] = float(all_conf.mean()) if len(all_conf) > 0 else 0.0
    agg['n_folds'] = successful_folds

    print(f"\n  {dataset_name} — AGGREGATE ({successful_folds} folds): ACC={agg['accuracy']:.4f}  "
          f"F1={agg['f1']:.4f}  AUC={agg['roc_auc']:.4f}")

    return agg, fold_metrics, best_state, (all_true, all_prob, all_conf, all_subj)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='SSVB-CASA-AIS Production Training')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to train on: stressid, wesad, empathicschool, combined')
    parser.add_argument('--model-type', '--model_type', type=str, default=None,
                        choices=['ssvb', 'conv_moe_mf', 'cnn_baseline', 'cnn_baseline_grl'],
                        help='Model type: ssvb, conv_moe_mf, cnn_baseline, cnn_baseline_grl')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate setup without full training')
    args = parser.parse_args()

    if args.model_type:
        CONFIG['model_type'] = args.model_type

    if args.dataset:
        datasets = [args.dataset]
    else:
        # Default: all datasets with enriched data available
        available = ['stressid', 'wesad', 'empathicschool']
        datasets = [d for d in available if os.path.isdir(os.path.join(ENRICHED_DIR, d))]
        if not datasets:
            print("  ERROR: No enriched datasets found. Run build_enriched_training_data.py first.")
            sys.exit(1)

    print(f"{'='*60}")
    print(f"  SSVB-CASA-AIS Production Training Pipeline")
    print(f"{'='*60}")
    print(f"  Config: {json.dumps(CONFIG, indent=2)}")
    print(f"  Datasets: {datasets}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"  Deploy: {DEPLOY_DIR}")

    # Validate enriched data
    print(f"\n{'-'*60}")
    print("  Validating enriched datasets...")
    load_all_datasets(datasets)
    print("  All datasets validated.")

    if args.dry_run:
        print("\n  Dry-run: validating SSVBDataset...")
        ds = SSVBDataset(datasets[0], seq_len=30, augment=False)
        sample = ds[0]
        print(f"  Sample length: {len(sample)} (expected 13: 9 feats + label + subj_id + dataset_id + weight)")
        for i in range(9):
            print(f"    Feat[{i}]: shape={sample[i].shape}")
        print(f"  Label:     {sample[-4]}")
        print(f"  Subj ID:   {sample[-3]}")
        print(f"  Dataset ID:{sample[-2]}")
        print(f"  Weight:    {sample[-1]}")
        print("  Dry-run complete — no training executed.")
        return

    # Train per dataset (deduplicate to avoid training combined twice)
    all_results = {}
    combined_true, combined_prob, combined_conf = [], [], []
    combined_subj_labels = []
    combined_ds_labels = []
    best_thresh_for_export = 0.5
    combined_metrics_at_opt = None

    seen_datasets = set()
    for ds_name in datasets + ['combined']:
        if ds_name in seen_datasets:
            continue
        seen_datasets.add(ds_name)

        # Skip combined if 'combined' not in enriched dir (built separately)
        if ds_name == 'combined':
            combined_path = os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')
            if not os.path.exists(combined_path):
                print(f"\n  SKIP combined: enriched combined data not found at {combined_path}")
                continue

        meta_path = os.path.join(ENRICHED_DIR, ds_name, 'metadata.parquet')
        meta = pd.read_parquet(meta_path)
        n_subj = meta['subject_id'].nunique()
        n_rows = len(meta)

        if n_subj < 2:
            print(f"\n  SKIP {ds_name}: only {n_subj} subjects (need ≥2 for LOSO)")
            continue

        print(f"\n{'='*60}")
        print(f"  Training: {ds_name.upper()} ({n_rows} windows, {n_subj} subjects)")
        print(f"{'='*60}")

        CONFIG['trained_on'] = ds_name
        CONFIG['n_subjects'] = n_subj

        model_type_str = CONFIG.get('model_type', 'ssvb')
        agg, fold_metrics, best_state, (y_true, y_prob, y_conf, y_subj) = \
            run_cross_validation(ds_name, CONFIG)

        if agg is None:
            print(f"\n  SKIP {ds_name}: cross-validation returned no results")
            continue

        all_results[ds_name] = {
            'aggregate': agg,
            'folds': fold_metrics,
        }
        combined_true.append(y_true)
        combined_prob.append(y_prob)
        combined_conf.append(y_conf)
        combined_subj_labels.append(np.hstack(y_subj))
        combined_ds_labels.extend([ds_name] * len(y_true))

        # Find optimal threshold for combined dataset
        if ds_name == 'combined':
            best_thresh_for_export, thresh_results = find_optimal_threshold(y_true, y_prob, metric='f1')
            combined_metrics_at_opt = calculate_metrics(y_true, y_prob, threshold=best_thresh_for_export)
            print(f"\n  Optimal threshold (max F1): {best_thresh_for_export:.3f}")
            print(f"    At optimal: prec={combined_metrics_at_opt['precision']:.4f}  recall={combined_metrics_at_opt['recall']:.4f}  f1={combined_metrics_at_opt['f1']:.4f}")
            for r in thresh_results:
                if abs(r['threshold'] - 0.5) < 0.01 or abs(r['threshold'] - best_thresh_for_export) < 0.01 or \
                   abs(r['threshold'] - 0.2) < 0.01 or abs(r['threshold'] - 0.32) < 0.01:
                    print(f"    thresh={r['threshold']:.2f}  prec={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")
        else:
            best_thresh_for_export = 0.5
            combined_metrics_at_opt = None

        # Per-model directory
        model_dir = os.path.join(REPORTS_DIR, model_type_str, ds_name)
        os.makedirs(model_dir, exist_ok=True)
        generate_plots(y_true, y_prob, f'{model_type_str} ({ds_name})', model_dir, threshold=best_thresh_for_export)

        # Save per-fold metrics as CSV
        fold_rows = []
        for fi, fm in enumerate(fold_metrics):
            row = {'fold': fi + 1}
            row.update(fm)
            fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(os.path.join(model_dir, 'fold_metrics.csv'), index=False)

        # Save aggregate metrics as JSON
        agg_save = dict(agg)
        if ds_name == 'combined':
            agg_save['optimal_threshold'] = best_thresh_for_export
            agg_save['metrics_at_optimal'] = combined_metrics_at_opt
        agg_path = os.path.join(model_dir, 'aggregate_metrics.json')
        with open(agg_path, 'w') as f:
            json.dump({'aggregate': agg_save, 'model_type': model_type_str,
                       'dataset': ds_name, 'n_subjects': n_subj}, f, indent=2)

        # Save per-subject predictions CSV
        pred_df = pd.DataFrame({
            'true': y_true, 'prob': y_prob, 'confidence': y_conf,
            'subject_id': np.hstack(y_subj),
            'dataset': [ds_name] * len(y_true),
        })
        pred_df.to_csv(os.path.join(model_dir, 'predictions.csv'), index=False)

        # Save checkpoint
        if best_state is not None:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'{ds_name}_{model_type_str}_best.pt')
            if CONFIG.get('model_type') == 'conv_moe_mf':
                m = ConvMoE_MF(hidden_dim=CONFIG['hidden_dim'], embed_dim=8,
                               num_subjects=n_subj, num_datasets=3,
                               grl_alpha_subj=CONFIG.get('grl_alpha_subj', 0.02),
                               grl_alpha_ds=CONFIG.get('grl_alpha_ds', 0.02))
            elif CONFIG.get('model_type') == 'cnn_baseline':
                m = CNNBaseline(total_feat_dim=69, num_subjects=n_subj)
            elif CONFIG.get('model_type') == 'cnn_baseline_grl':
                m = CNNBaselineGRL(total_feat_dim=69, num_subjects=n_subj,
                                   grl_alpha=CONFIG.get('grl_alpha_subj', 0.02))
            else:
                m = SSVBCASA_AIS(hidden_dim=CONFIG['hidden_dim'],
                                 num_subjects=n_subj)
            m.load_state_dict(best_state)
            save_checkpoint(m, None, 0, agg, ckpt_path)

        # Export combined model weights (per-model naming)
        if ds_name == 'combined':
            model_tag = CONFIG.get('model_type', 'ssvb')
            deploy_path = os.path.join(DEPLOY_DIR, f'ssvb_casa_ais_production_{model_tag}.pt')
            if best_state is not None:
                if CONFIG.get('model_type') == 'conv_moe_mf':
                    m = ConvMoE_MF(hidden_dim=CONFIG['hidden_dim'], embed_dim=8,
                                   num_subjects=n_subj, num_datasets=3,
                                   grl_alpha_subj=CONFIG.get('grl_alpha_subj', 0.02),
                                   grl_alpha_ds=CONFIG.get('grl_alpha_ds', 0.02))
                elif CONFIG.get('model_type') == 'cnn_baseline':
                    m = CNNBaseline(total_feat_dim=69, num_subjects=n_subj)
                elif CONFIG.get('model_type') == 'cnn_baseline_grl':
                    m = CNNBaselineGRL(total_feat_dim=69, num_subjects=n_subj,
                                       grl_alpha=CONFIG.get('grl_alpha_subj', 0.02))
                else:
                    m = SSVBCASA_AIS(hidden_dim=CONFIG['hidden_dim'],
                                     num_subjects=n_subj)
                m.load_state_dict(best_state)
                torch.save(m.state_dict(), deploy_path)
                export_deployment_weights(m, deploy_path, optimal_threshold=best_thresh_for_export)
                print(f"\n  [DEPLOY] Production weights saved to {deploy_path}")

    # Build summary report
    combined_true = np.hstack(combined_true)
    combined_prob = np.hstack(combined_prob)
    combined_conf = np.hstack(combined_conf)

    combined_metrics_default = calculate_metrics(combined_true, combined_prob)
    combined_metrics_at_opt = calculate_metrics(combined_true, combined_prob, threshold=best_thresh_for_export)
    combined_metrics = combined_metrics_default
    combined_metrics['optimal'] = combined_metrics_at_opt
    combined_metrics['mean_confidence'] = float(combined_conf.mean())
    combined_metrics['optimal_threshold'] = best_thresh_for_export

    combined_subj_labels = np.hstack(combined_subj_labels)
    results_df = pd.DataFrame({
        'true': combined_true,
        'pred': (combined_prob >= best_thresh_for_export).astype(int),
        'prob': combined_prob,
        'subject_id': combined_subj_labels,
        'dataset': combined_ds_labels,
    })
    subj_metrics = per_subject_metrics(results_df)
    ds_metrics = per_dataset_metrics(results_df)

    # Write final report
    report = {
        'config': CONFIG,
        'per_dataset': {k: v['aggregate'] for k, v in all_results.items()},
        'combined': combined_metrics,
        'per_subject': subj_metrics,
        'per_dataset_breakdown': ds_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save plots for combined (in per-model dir)
    model_type_str = CONFIG.get('model_type', 'ssvb')
    combined_dir = os.path.join(REPORTS_DIR, model_type_str, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    generate_plots(combined_true, combined_prob,
                   f'{model_type_str} (All Datasets)', combined_dir,
                   threshold=best_thresh_for_export)

    report_path = os.path.join(combined_dir, 'metrics.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    model_tag = CONFIG.get('model_type', 'ssvb')
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — {model_tag}")
    print(f"{'='*60}")
    for ds_name, res in all_results.items():
        a = res['aggregate']
        print(f"  {ds_name:20s}  ACC={a['accuracy']:.4f}  F1={a['f1']:.4f}  "
              f"AUC={a['roc_auc']:.4f}  Conf={a.get('mean_confidence', 0):.4f}")
    print(f"  {'combined':20s}  ACC={combined_metrics['accuracy']:.4f}  "
          f"F1={combined_metrics['f1']:.4f}  AUC={combined_metrics['roc_auc']:.4f}")
    print(f"\n  Per-subject accuracy: mean={subj_metrics['subject_acc_mean']:.4f} "
          f"std={subj_metrics['subject_acc_std']:.4f}")
    print(f"  Per-dataset: {ds_metrics}")
    print(f"\n  Reports: {combined_dir}")
    print(f"  Weights: {DEPLOY_DIR}/ssvb_casa_ais_production_{model_tag}.pt")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
