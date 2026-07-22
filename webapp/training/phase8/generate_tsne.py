"""
generate_tsne.py — Generates t-SNE 2D visualizations of ConvMoE-MF fused embeddings.

Plots:
1. Embeddings colored by Dataset (StressID, WESAD, EmpathicSchool)
2. Embeddings colored by Stress Label (0=Calm, 1=Stressed)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'webapp'))

from backend.runtime.conv_moe_mf import ConvMoE_MF
from train_ssvb_production import SSVBDataset, _unpack_batch, CONFIG

ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'Phase_3_Production',
                           'production_model', 'ssvb_casa_ais_production')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'webapp', 'backend', 'runtime', 'models', 'ssvb_casa_ais_production.pt')

def main():
    print("=" * 60)
    print("  Generating t-SNE Visualization for ConvMoE-MF Embeddings")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    ds_path = os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')
    if not os.path.exists(ds_path):
        print(f"Error: {ds_path} not found")
        return

    meta = pd.read_parquet(ds_path)
    n_subj = meta['subject_id'].nunique()
    ds = SSVBDataset('combined', seq_len=5, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # Load model
    model = ConvMoE_MF(hidden_dim=16, embed_dim=8, num_subjects=n_subj, num_datasets=3).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found, using initialized model")
    model.eval()

    # Extract embeddings
    all_embeddings = []
    all_labels = []
    all_datasets = []

    print("Extracting fused embeddings...")
    with torch.no_grad():
        for batch in loader:
            eye, mouth, gface, sp, mfcc, qual, card, eda, soma, label, subj_id, dataset_id, _ = _unpack_batch(batch, device)
            out = model(eye, mouth, gface, sp, mfcc, qual, card, eda, soma, return_all=True)
            fused = out['fused_embedding'].cpu().numpy()
            all_embeddings.append(fused)
            all_labels.append(label.cpu().numpy())
            all_datasets.append(dataset_id.cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.hstack(all_labels)
    datasets = np.hstack(all_datasets)

    # Subsample 5,000 points for fast, clean t-SNE plotting
    if len(embeddings) > 5000:
        idx = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        datasets = datasets[idx]

    print(f"Running t-SNE on {len(embeddings)} sampled embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedding_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Plot 1: Color by Dataset
    ds_names = {0: 'StressID', 1: 'WESAD', 2: 'EmpathicSchool'}
    colors = {0: '#2b5c8f', 1: '#d95f02', 2: '#7570b3'}
    for d_id, d_name in ds_names.items():
        mask = (datasets == d_id)
        if mask.any():
            axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=colors[d_id], label=d_name, alpha=0.6, s=12)
    axes[0].set_title('ConvMoE-MF Fused Embeddings (by Dataset)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')

    # Plot 2: Color by Stress Label
    stress_colors = {0: '#2ca02c', 1: '#d62728'}
    stress_names = {0: 'Calm / Baseline (0)', 1: 'Stressed (1)'}
    for l_id, l_name in stress_names.items():
        mask = (labels == l_id)
        if mask.any():
            axes[1].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                            c=stress_colors[l_id], label=l_name, alpha=0.6, s=12)
    axes[1].set_title('ConvMoE-MF Fused Embeddings (by Stress Label)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')

    plt.tight_layout()
    tsne_path = os.path.join(REPORTS_DIR, 'tsne.png')
    plt.savefig(tsne_path, bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved to: {tsne_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()
