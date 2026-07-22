"""
Sweep dataset-GRL alpha to find optimal suppression of zero-pattern shortcut.

Tests grl_alpha_ds in {0.02, 0.05, 0.10, 0.20} on combined dataset.
Reports: (1) stress classification AUC, (2) dataset prediction AUC
Lower dataset AUC = better suppression. Higher stress AUC = better accuracy.
"""
import os, sys, json, copy, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

parser = argparse.ArgumentParser(description='Sweep dataset-GRL alpha')
parser.add_argument('--dry-run', action='store_true', help='Validate setup without running sweep')
args = parser.parse_args()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'webapp'))

from backend.runtime.conv_moe_mf import ConvMoE_MF
from train_ssvb_production import (SSVBDataset, _unpack_batch,
    train_supervised_epoch, evaluate, CONFIG)

ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'Phase_3_Production',
                           'production_model', 'ssvb_casa_ais_production',
                           'sweep_results')
os.makedirs(REPORTS_DIR, exist_ok=True)

device = torch.device(CONFIG['device'])
print(f"Device: {device}")

if args.dry_run:
    print("  Dry-run mode: sweep_grl_alpha setup validated.")
    sys.exit(0)

sweep_config = CONFIG.copy()
sweep_config['ssl_epochs'] = 2
sweep_config['ft_epochs'] = 4
sweep_config['batch_size'] = 256
sweep_config['model_type'] = 'conv_moe_mf'

ALPHAS = [0.02, 0.05, 0.10, 0.20]
DATASET = 'combined'
results = []

for alpha_ds in ALPHAS:
    print(f"\n{'='*60}")
    print(f"  Sweep: grl_alpha_ds = {alpha_ds}")
    print(f"{'='*60}")

    sweep_config['grl_alpha_ds'] = alpha_ds
    sweep_config['grl_alpha_subj'] = 0.02

    meta_path = os.path.join(ENRICHED_DIR, DATASET, "metadata.parquet")
    meta = pd.read_parquet(meta_path)
    subjects = sorted(meta['subject_id'].unique())
    rng = np.random.RandomState(sweep_config['seed'])
    rng.shuffle(subjects)

    n_folds = min(3, len(subjects))
    fold_stress_aucs = []
    fold_ds_aucs = []
    fold_stress_accs = []

    for fold in range(n_folds):
        test_subj = subjects[fold]
        train_subjs = [s for s in subjects if s != test_subj]
        print(f"  Fold {fold+1}/{n_folds} (test: {test_subj})")

        train_idx = meta[meta['subject_id'].isin(train_subjs)].index.values
        test_idx  = meta[meta['subject_id'] == test_subj].index.values

        ds_weight_map = {}
        for dn in meta['dataset'].unique() if 'dataset' in meta.columns else [DATASET]:
            key = f'dataset_weight_{dn}'
            ds_weight_map[dn] = sweep_config.get(key, 1.0)

        class IndexedDataset(Dataset):
            def __init__(self, base, indices):
                self.base = base; self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]

        full_ds = SSVBDataset(DATASET, seq_len=sweep_config['seq_len'],
                              augment=False, dataset_weights=ds_weight_map)
        train_ds = IndexedDataset(SSVBDataset(DATASET, seq_len=sweep_config['seq_len'],
                                  augment=True, noise_std=sweep_config['noise_std'],
                                  modality_dropout=sweep_config['modality_dropout'],
                                  dataset_weights=ds_weight_map), train_idx)
        test_ds = IndexedDataset(full_ds, test_idx)

        train_loader = DataLoader(train_ds, batch_size=sweep_config['batch_size'],
                                  shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=sweep_config['batch_size'],
                                 shuffle=False, num_workers=0)

        num_datasets = len(meta['dataset'].unique()) if 'dataset' in meta.columns else 1
        model = ConvMoE_MF(hidden_dim=sweep_config['hidden_dim'], embed_dim=8,
                           num_subjects=len(subjects),
                           num_datasets=num_datasets,
                           grl_alpha_subj=sweep_config['grl_alpha_subj'],
                           grl_alpha_ds=sweep_config['grl_alpha_ds']).to(device)

        criterion_subj = torch.nn.CrossEntropyLoss()
        criterion_ds = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=sweep_config['lr_ft'],
                                weight_decay=sweep_config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sweep_config['ft_epochs'])

        for ep in range(sweep_config['ft_epochs']):
            loss = train_supervised_epoch(model, train_loader, opt,
                                          criterion_subj, criterion_ds, device, sweep_config)
            scheduler.step()

        val_prob, val_true, val_conf, val_subj = evaluate(model, test_loader, device)
        stress_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.5
        stress_acc = accuracy_score(val_true, (val_prob >= 0.5).astype(int))

        # Measure dataset leakage
        model.eval()
        ds_true_all = []
        ds_prob_all = []
        with torch.no_grad():
            for batch in test_loader:
                eye, mouth, gface, sp, mfcc, qual, card, eda, soma, _, _, ds_id, _ = _unpack_batch(batch, device)
                out = model(eye, mouth, gface, sp, mfcc, qual, card, eda, soma, return_all=True)
                ds_probs = torch.softmax(out['dataset_logits'], dim=1).cpu().numpy()
                ds_true_all.append(ds_id.cpu().numpy())
                ds_prob_all.append(ds_probs)
        ds_true_all = np.hstack(ds_true_all)
        ds_prob_all = np.vstack(ds_prob_all)

        ds_aucs = []
        for c in range(num_datasets):
            if len(np.unique(ds_true_all == c)) > 1:
                auc = roc_auc_score((ds_true_all == c).astype(int), ds_prob_all[:, c])
                ds_aucs.append(auc)
        ds_auc = np.mean(ds_aucs) if ds_aucs else 0.5

        print(f"    Fold {fold+1}: stress AUC={stress_auc:.4f}, stress ACC={stress_acc:.4f}, "
              f"dataset AUC={ds_auc:.4f}")

        fold_stress_aucs.append(stress_auc)
        fold_ds_aucs.append(ds_auc)
        fold_stress_accs.append(stress_acc)

    avg_stress_auc = np.mean(fold_stress_aucs)
    avg_ds_auc = np.mean(fold_ds_aucs)
    avg_stress_acc = np.mean(fold_stress_accs)
    gap = avg_stress_auc - avg_ds_auc

    print(f"  -> Alpha={alpha_ds:.2f}: stress AUC={avg_stress_auc:.4f}, "
          f"stress ACC={avg_stress_acc:.4f}, dataset AUC={avg_ds_auc:.4f}, gap={gap:+.4f}")

    results.append({
        'grl_alpha_ds': alpha_ds,
        'stress_auc': float(avg_stress_auc),
        'stress_acc': float(avg_stress_acc),
        'dataset_auc': float(avg_ds_auc),
        'fold_stress_aucs': [float(x) for x in fold_stress_aucs],
        'fold_ds_aucs': [float(x) for x in fold_ds_aucs],
    })

print(f"\n{'='*60}")
print(f"  SWEEP RESULTS")
print(f"{'='*60}")
print(f"  {'Alpha':>6s}  {'Stress AUC':>10s}  {'Stress ACC':>10s}  {'Dataset AUC':>11s}  {'Gap':>8s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*8}")
for r in results:
    gap = r['stress_auc'] - r['dataset_auc']
    print(f"  {r['grl_alpha_ds']:>6.2f}  {r['stress_auc']:>10.4f}  {r['stress_acc']:>10.4f}  {r['dataset_auc']:>11.4f}  {gap:>+8.4f}")

report_path = os.path.join(REPORTS_DIR, 'sweep_results.json')
with open(report_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved: {report_path}")

best = max(results, key=lambda r: r['stress_auc'] - r['dataset_auc'])
print(f"\n  Best alpha (max gap): {best['grl_alpha_ds']:.2f}")
