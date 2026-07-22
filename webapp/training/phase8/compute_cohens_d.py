"""
compute_cohens_d.py — WS2: Empirically validate cross-dataset convergence.

Computes Cohen's d for core physiological channels per dataset.
Vectorized (loads all windows at once, not per-row loop).

Usage:
    python compute_cohens_d.py
    python compute_cohens_d.py --dataset stressid
    python compute_cohens_d.py --save-json
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

ENRICHED_DIR = r'C:\Users\StressProject.DESKTOP-U6P7JQT\Desktop\StressDetectionUsingML\data\enriched_training_data'

PHYSIO_CHANNELS = [
    'hr', 'hrv_rmssd', 'eda_clean', 'eda_phasic',
    'scr_count', 'resp_rate',
]

# Physio indices in the 69-D concatenated feature vector
# physio_cardio (33-34): hr, hrv_rmssd
# physio_eda   (35-37): eda_clean, eda_tonic, eda_phasic
# physio_somatic (38-45): scr_count, resp_rate, resp_amplitude, ...
CHANNEL_INDICES = [33, 34, 35, 37, 38, 39]


def load_dataset_physio(dataset_name):
    """Load all physio features for a dataset, mean-pooled over time.
    Returns (pooled_feats, labels) where pooled_feats is [N, 13].
    """
    seq_path = os.path.join(ENRICHED_DIR, dataset_name, "sequences.npz")
    meta_path = os.path.join(ENRICHED_DIR, dataset_name, "metadata.parquet")
    if not os.path.exists(seq_path):
        return None, None

    features = np.load(seq_path)
    meta = pd.read_parquet(meta_path)

    # Concatenate all 9 groups into [N, T=30, 69]
    all_feats = np.concatenate([features[k] for k in sorted(features.keys())], axis=-1)
    # Mean-pool over time → [N, 69]
    pooled = np.nanmean(all_feats, axis=1)
    # Extract specific physio channels by index
    physio = pooled[:, CHANNEL_INDICES]  # [N, 6]
    labels = meta['label'].values

    return physio, labels


def compute_cohens_d(stressed, calm):
    if len(stressed) < 2 or len(calm) < 2:
        return None
    pooled_std = np.sqrt((np.nanstd(stressed, ddof=1) ** 2 +
                          np.nanstd(calm, ddof=1) ** 2) / 2)
    if pooled_std < 1e-8:
        return None
    return float((np.nanmean(stressed) - np.nanmean(calm)) / pooled_std)


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's d per dataset")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--save-json', action='store_true')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ['stressid', 'wesad', 'empathicschool', 'combined']
    all_results = {}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds.upper()}")
        print(f"{'='*60}")
        physio, labels = load_dataset_physio(ds)
        if physio is None:
            print(f"  SKIP: enriched data not found at {ENRICHED_DIR}/{ds}")
            continue

        n_stress = int((labels == 1).sum())
        n_calm = int((labels == 0).sum())
        print(f"  Windows: {len(labels)} total ({n_stress} stress, {n_calm} calm)")

        results = {}
        for j, ch in enumerate(PHYSIO_CHANNELS):
            vals = physio[:, j]
            stressed = vals[labels == 1]
            calm = vals[labels == 0]

            # Filter NaN/inf
            stressed = stressed[np.isfinite(stressed)]
            calm = calm[np.isfinite(calm)]

            d = compute_cohens_d(stressed, calm)
            if d is not None:
                results[ch] = {
                    'cohens_d': round(d, 4),
                    'stressed_mean': round(float(np.nanmean(stressed)), 4),
                    'stressed_std': round(float(np.nanstd(stressed, ddof=1)), 4),
                    'calm_mean': round(float(np.nanmean(calm)), 4),
                    'calm_std': round(float(np.nanstd(calm, ddof=1)), 4),
                    'n_stressed': int(len(stressed)),
                    'n_calm': int(len(calm)),
                }
            else:
                results[ch] = None

        all_results[ds] = results

        # Print table
        print(f"\n  {'Channel':20s}  {'d':>8s}  {'Stress M':>10s}  {'Calm M':>10s}  {'N_s/N_c':>10s}")
        print(f"  {'-'*60}")
        for ch in PHYSIO_CHANNELS:
            r = results.get(ch)
            if r:
                arrow = "↑" if r['cohens_d'] > 0 else "↓"
                print(f"  {ch:20s}  {r['cohens_d']:>+8.4f}  {r['stressed_mean']:>10.4f}  "
                      f"{r['calm_mean']:>10.4f}  {r['n_stressed']}/{r['n_calm']}")
            else:
                print(f"  {ch:20s}  {'N/A':>8s}  {'N/A':>10s}  {'N/A':>10s}  {'N/A':>10s}")

    # Direction agreement across per-dataset results (exclude 'combined' aggregate)
    per_ds_results = {k: v for k, v in all_results.items() if k != 'combined'}
    if len(per_ds_results) >= 2:
        print(f"\n{'='*60}")
        print("  DIRECTION AGREEMENT (per-dataset, excluding combined aggregate)")
        print(f"{'='*60}")
        for ch in PHYSIO_CHANNELS:
            signs = set()
            for ds, res in per_ds_results.items():
                r = res.get(ch)
                if r and r['cohens_d'] != 0:
                    signs.add(int(np.sign(r['cohens_d'])))
            if len(signs) == 1:
                sgn = list(signs)[0]
                print(f"  {ch:20s} [OK] ALL AGREE (d {sgn:+.0f})")
            elif len(signs) > 1:
                print(f"  {ch:20s} [CONFLICT] signs={signs}")
            else:
                print(f"  {ch:20s} [NODATA]")

    if args.save_json:
        out_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'cohens_d_results.json')
        import json as j
        with open(out_path, 'w') as f:
            j.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
