"""
Phase 3: Train CNNBaselineGRL with full analysis pipeline.
Saves per-window predictions, computes bootstrap CIs, ROC/PR curves, etc.
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
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.calibration import calibration_curve
# bootstrap CI computed via custom function

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase3_production'))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

from phase3_production.train import (SSVBDataset, CNNBaseline, CNNBaselineGRL,
    CONFIG as TRAIN_CONFIG, _unpack_batch, calculate_metrics,
    find_optimal_threshold, per_subject_metrics, per_dataset_metrics,
    per_source_dataset_metrics, contrastive_loss, generate_plots)

ENRICHED_DIR = str(PROJECT_ROOT / 'data' / 'enriched_training_data')
REPORTS_DIR  = str(PROJECT_ROOT / 'phase3_results')
os.makedirs(REPORTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

GROUP_KEYS = ['face_eye', 'face_global_face', 'face_mouth',
              'physio_cardio', 'physio_eda', 'physio_somatic',
              'voice_mfcc', 'voice_quality', 'voice_spectral_prosody']


def compute_bootstrap_ci(y_true, y_prob, metric_fn, n_resamples=2000, ci=95):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)
    return float(lower), float(upper), float(np.mean(scores)), float(np.std(scores))


def compute_bootstrap_delta(y_true, y_prob_a, y_prob_b, metric_fn, n_resamples=2000):
    """Compute bootstrap CI for the difference between two models."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas.append(metric_fn(y_true[idx], y_prob_a[idx]) - metric_fn(y_true[idx], y_prob_b[idx]))
    deltas = np.array(deltas)
    lower = np.percentile(deltas, 2.5)
    upper = np.percentile(deltas, 97.5)
    return float(lower), float(upper), float(np.mean(deltas)), float(np.std(deltas))


def save_predictions(dataset_name, all_subjects, all_true, all_prob, all_conf, save_dir):
    """Save per-window predictions for bootstrap analysis."""
    df = pd.DataFrame({
        'subject_id': all_subjects,
        'true': all_true,
        'prob': all_prob,
        'confidence': all_conf,
        'dataset': dataset_name,
    })
    path = os.path.join(save_dir, f'predictions_{dataset_name}.csv')
    df.to_csv(path, index=False)
    print(f"    Saved predictions: {path} ({len(df)} windows)")
    return df


def run_benchmark_with_predictions(
    dataset_name, model_entry, config, device,
    exclude_dataset=None, exclude_subjects=None,
    save_preds=True):
    """
    Run LOSO benchmark on enriched dataset.
    Saves per-window predictions + computes point metrics.
    Returns: agg_metrics, fold_metrics_list, predictions_df
    """
    data_dir = os.path.join(ENRICHED_DIR, dataset_name)
    if not os.path.exists(data_dir):
        print(f"  SKIP {dataset_name}: data not found")
        return None, [], None

    meta = pd.read_parquet(os.path.join(data_dir, 'metadata.parquet'))
    group_dims_path = os.path.join(data_dir, 'group_dims.json')
    with open(group_dims_path) as f:
        group_dims = json.load(f)

    # Exclusions
    if exclude_dataset and 'dataset' in meta.columns:
        meta = meta[meta['dataset'] != exclude_dataset]
    if exclude_subjects:
        meta = meta[~meta['subject_id'].isin(exclude_subjects)]

    n_subjects = meta['subject_id'].nunique()
    n_datasets = meta['dataset'].nunique() if 'dataset' in meta.columns else 1
    subjects = sorted(meta['subject_id'].unique())
    total_windows = len(meta)

    print(f"\n    Dataset: {dataset_name} ({total_windows} windows, {n_subjects} subjects)")

    # Skip single-class subjects
    subj_label_set = meta.groupby('subject_id')['label'].unique()
    valid_subjects = [s for s in subjects if len(subj_label_set[s]) > 1]
    skipped = [s for s in subjects if s not in valid_subjects]

    selected = valid_subjects
    if len(selected) == 0:
        return None, [], None

    # Subject mapping
    subj_to_idx = {s: i for i, s in enumerate(sorted(meta['subject_id'].unique()))}
    idx_to_subj = {i: s for s, i in subj_to_idx.items()}
    train_subjects = set(meta['subject_id'].unique())

    # Build model (use original from benchmark)
    from scripts.run_all_models_benchmark import build_model as orig_build_model
    from scripts.run_all_models_benchmark import forward_model as orig_forward_model

    # DataLoader wrappers
    class IndexedDataset(Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    ds_weight_map = {}
    for ds_name in meta['dataset'].unique() if 'dataset' in meta.columns else [dataset_name]:
        key = f'dataset_weight_{ds_name}'
        ds_weight_map[ds_name] = config.get(key, 1.0)

    all_true, all_prob, all_conf, all_subj = [], [], [], []
    fold_metrics_list = []
    successful_folds = 0

    for fold, test_subj in enumerate(selected, 1):
        test_labels = meta[meta['subject_id'] == test_subj]['label'].values
        if len(np.unique(test_labels)) < 2:
            continue

        print(f"\n      Fold {fold}/{len(selected)} (test: {test_subj})")

        train_subjs = [s for s in subjects if s != test_subj]
        train_idx = meta[meta['subject_id'].isin(train_subjs)].index.values
        test_idx = meta[meta['subject_id'] == test_subj].index.values

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
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

        model = orig_build_model(model_entry, group_dims, n_subjects, n_datasets, device)
        criterion_subj = nn.CrossEntropyLoss()

        # SSL pretraining (only models with explicit experts)
        has_experts = hasattr(model, 'exp_eye') or hasattr(model, 'enc_face')
        if config['ssl_epochs'] > 0 and has_experts:
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

        # Fine-tuning
        opt_ft = optim.AdamW(model.parameters(), lr=model_entry.learning_rate,
                             weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=config['ft_epochs'])
        best_fold_auc = 0.0

        for ep in range(config['ft_epochs']):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                logits, subj_logits, confidence, label, subj_id = \
                    orig_forward_model(model, model_entry, batch, device)
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
            val_probs, val_true = [], []
            with torch.no_grad():
                for batch in test_loader:
                    logits, _, _, label, _ = orig_forward_model(model, model_entry, batch, device)
                    val_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    val_true.append(label.cpu().numpy())
            val_probs = np.hstack(val_probs)
            val_true = np.hstack(val_true)
            val_auc = roc_auc_score(val_true, val_probs) if len(np.unique(val_true)) > 1 else 0.5
            scheduler.step()

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc

        # Final fold evaluation
        model.eval()
        fold_prob, fold_true, fold_conf = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, _, confidence, label, _ = orig_forward_model(model, model_entry, batch, device)
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
        all_subj.append(np.array([test_subj] * len(fold_true)))
        successful_folds += 1
        print(f"      -> Fold {fold}: ACC={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    if successful_folds == 0:
        return None, [], None

    all_true = np.hstack(all_true)
    all_prob = np.hstack(all_prob)
    all_conf = np.hstack(all_conf)
    all_subj_h = np.hstack(all_subj)

    agg = calculate_metrics(all_true, all_prob)
    agg['mean_confidence'] = float(all_conf.mean())
    agg['n_folds'] = successful_folds
    print(f"\n      {dataset_name}: ACC={agg['accuracy']:.4f}  F1={agg['f1']:.4f}  AUC={agg['roc_auc']:.4f}")

    # Save predictions
    preds_df = None
    save_dir = os.path.join(REPORTS_DIR, model_entry.name)
    os.makedirs(save_dir, exist_ok=True)
    if save_preds:
        preds_df = save_predictions(dataset_name, all_subj_h, all_true, all_prob, all_conf, save_dir)

    return agg, fold_metrics_list, {'true': all_true, 'prob': all_prob, 'conf': all_conf, 'subjects': all_subj_h}


def analyze_results(dataset_name, preds_dict, model_name, save_dir):
    """Compute bootstrap CIs, ROC/PR curves, confusion matrices, calibration."""
    y_true = preds_dict['true']
    y_prob = preds_dict['prob']
    y_conf = preds_dict['conf']

    # 1. Bootstrap CIs
    print(f"\n    Computing bootstrap CIs ({dataset_name})...")
    auc_fn = lambda y, p: roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
    f1_fn = lambda y, p: f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    acc_fn = lambda y, p: accuracy_score(y, (p >= 0.5).astype(int))

    ci_auc = compute_bootstrap_ci(y_true, y_prob, auc_fn)
    ci_f1 = compute_bootstrap_ci(y_true, y_prob, f1_fn)
    ci_acc = compute_bootstrap_ci(y_true, y_prob, acc_fn)

    print(f"      AUC: {np.mean(y_prob[y_true==1]) - np.mean(y_prob[y_true==0]):.4f}  "
          f"CI: [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]  mean={ci_auc[2]:.4f}")
    print(f"      F1:  CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]  mean={ci_f1[2]:.4f}")
    print(f"      ACC: CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]  mean={ci_acc[2]:.4f}")

    # 2. Optimal threshold analysis
    opt_thresh, thresh_results = find_optimal_threshold(y_true, y_prob, metric='f1')
    y_pred_opt = (y_prob >= opt_thresh).astype(int)
    f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
    print(f"      Optimal threshold: {opt_thresh:.3f} (F1={f1_opt:.4f})")

    # 3. ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thresholds.tolist()}

    # 4. PR curve data
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_data = {'precisions': precisions.tolist(), 'recalls': recalls.tolist(),
               'thresholds': pr_thresholds.tolist()}

    # 5. Confusion matrices
    cm_05 = confusion_matrix(y_true, (y_prob >= 0.5).astype(int)).tolist()
    cm_opt = confusion_matrix(y_true, y_pred_opt).tolist()

    # 6. Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        cal_data = {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
    except Exception:
        cal_data = {}

    # 7. Subject-level metrics
    subj_metrics = {}
    df = pd.DataFrame({'subject_id': preds_dict['subjects'], 'true': y_true, 'prob': y_prob})
    for subj in sorted(df['subject_id'].unique()):
        s = df[df['subject_id'] == subj]
        if len(np.unique(s['true'])) > 1:
            subj_auc = roc_auc_score(s['true'], s['prob'])
        else:
            subj_auc = None
        subj_f1 = f1_score(s['true'], (s['prob'] >= 0.5).astype(int), zero_division=0)
        subj_metrics[str(subj)] = {
            'n_windows': len(s),
            'stress_ratio': float(s['true'].mean()),
            'auc': round(float(subj_auc), 4) if subj_auc is not None else None,
            'f1_0.5': round(float(subj_f1), 4),
        }

    # 8. Save analysis
    analysis = {
        'dataset': dataset_name,
        'model': model_name,
        'n_windows': len(y_true),
        'n_subjects': len(np.unique(preds_dict['subjects'])),
        'bootstrap_ci_95': {
            'auc': {'lower': round(ci_auc[0], 4), 'upper': round(ci_auc[1], 4),
                    'mean': round(ci_auc[2], 4), 'std': round(ci_auc[3], 4)},
            'f1':  {'lower': round(ci_f1[0], 4), 'upper': round(ci_f1[1], 4),
                    'mean': round(ci_f1[2], 4), 'std': round(ci_f1[3], 4)},
            'acc': {'lower': round(ci_acc[0], 4), 'upper': round(ci_acc[1], 4),
                    'mean': round(ci_acc[2], 4), 'std': round(ci_acc[3], 4)},
        },
        'optimal_threshold': {
            'threshold': round(float(opt_thresh), 4),
            'f1_at_optimal': round(float(f1_opt), 4),
        },
        'roc_curve': roc_data,
        'pr_curve': pr_data,
        'confusion_matrix_at_0.5': cm_05,
        'confusion_matrix_at_optimal': cm_opt,
        'calibration_curve': cal_data,
        'subject_metrics': subj_metrics,
        'mean_confidence': float(y_conf.mean()),
    }

    path = os.path.join(save_dir, f'analysis_{dataset_name}.json')
    with open(path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"    Saved analysis: {path}")
    return analysis


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['cnn_baseline_grl'],
                        help='Models to train (default: cnn_baseline_grl)')
    parser.add_argument('--datasets', nargs='+', default=['stressid', 'wesad', 'combined'],
                        help='Datasets to evaluate on')
    args = parser.parse_args()

    # Build model entries
    from scripts.run_all_models_benchmark import REGISTRY, ModelEntry, forward_model
    model_entries = [e for e in REGISTRY if e.name in args.models]
    if not model_entries:
        print(f"No models found for: {args.models}")
        return

    TRAIN_CFG = {
        'seq_len': 30, 'batch_size': 64, 'ssl_epochs': 5, 'ft_epochs': 20,
        'lr_ssl': 1e-3, 'weight_decay': 1e-4, 'noise_std': 0.05,
        'modality_dropout': 0.1, 'lambda_conf': 0.1, 'lambda_subj': 0.1,
    }

    for entry in model_entries:
        print(f"\n{'='*60}")
        print(f"  MODEL: {entry.name}")
        print(f"{'='*60}")

        save_dir = os.path.join(REPORTS_DIR, entry.name)
        os.makedirs(save_dir, exist_ok=True)

        all_predictions = {}
        all_analyses = {}

        for ds_name in args.datasets:
            agg, folds, preds = run_benchmark_with_predictions(
                ds_name, entry, TRAIN_CFG, DEVICE)
            if agg is None:
                continue
            all_predictions[ds_name] = agg

            # Full analysis with bootstrap CIs
            if preds is not None:
                analysis = analyze_results(ds_name, preds, entry.name, save_dir)
                all_analyses[ds_name] = analysis

        # Save summary
        summary = {
            'model': entry.name,
            'group': entry.group,
            'params': entry.init_kwargs,
            'per_dataset_summary': all_predictions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print results
        print(f"\n  {'='*60}")
        print(f"  RESULTS: {entry.name}")
        for ds_name, m in all_predictions.items():
            ds_analysis = all_analyses.get(ds_name, {})
            ci = ds_analysis.get('bootstrap_ci_95', {}).get('auc', {})
            ci_str = f" [95% CI: {ci.get('lower', '?'):.4f}-{ci.get('upper', '?'):.4f}]" if ci else ""
            opt = ds_analysis.get('optimal_threshold', {})
            opt_str = f"  optimal_thresh={opt.get('threshold', '?'):.3f}" if opt else ""
            print(f"    {ds_name:15s}: AUC={m['roc_auc']:.4f}{ci_str}  F1={m['f1']:.4f}{opt_str}")

    print(f"\n  Results: {REPORTS_DIR}")
    print(f"  Done.")


if __name__ == '__main__':
    main()
