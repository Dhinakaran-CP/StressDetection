"""
run.py — Phase 3 Production: Single entry point for all 4 models.

Runs SSVB-CASA-AIS, CNNBaseline, CNNBaseline+GRL, ConvMoE-MF sequentially
on the combined enriched dataset with 5-fold LOSO CV, threshold tuning,
and a comparison report.

Usage:
    .\venv\Scripts\python phase3_production\run.py
    .\venv\Scripts\python phase3_production\run.py --start-step 3
    .\venv\Scripts\python phase3_production\run.py --dry-run
"""
import os, sys, json, time, subprocess, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, '..', 'venv', 'Scripts', 'python.exe')
TRAIN_SCRIPT = os.path.join(ROOT, 'train.py')
RESULTS_DIR = os.path.join(ROOT, 'results')

STEPS = [
    (1, "SSVB-CASA-AIS (41K params, no GRL)",      'ssvb',             90),
    (2, "CNNBaseline (21K params, no GRL)",          'cnn_baseline',     90),
    (3, "CNNBaseline+GRL (23K params, subj-GRL)",    'cnn_baseline_grl', 90),
    (4, "ConvMoE-MF (8.8K params, dual GRL + conf)", 'conv_moe_mf',      90),
]


def run_step(step_num, step_name, model_tag, timeout_min):
    args = ['--dataset', 'combined', '--model_type', model_tag]
    cmd = [VENV_PY, TRAIN_SCRIPT] + args
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}/4: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=timeout_min*60)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  [OK] Step {step_num} completed in {elapsed:.1f}s")
    else:
        print(f"\n  [FAIL] Step {step_num} exited with code {result.returncode}")
    return result.returncode


def build_comparison():
    """Read all model results and produce comparison CSV + JSON."""
    import pandas as pd
    rows = []
    for num, name, tag, _ in STEPS:
        mp = os.path.join(RESULTS_DIR, tag, 'combined', 'aggregate_metrics.json')
        if not os.path.exists(mp):
            rows.append({'model': tag, 'accuracy': 'N/A', 'precision': 'N/A',
                         'recall': 'N/A', 'f1': 'N/A', 'auc_roc': 'N/A',
                         'avg_precision': 'N/A', 'opt_thresh': 'N/A'})
            continue
        with open(mp) as f:
            d = json.load(f)
        a = d.get('aggregate', {})
        rows.append({
            'model': tag,
            'accuracy':  f"{a.get('accuracy', 0):.4f}",
            'precision': f"{a.get('precision', 0):.4f}",
            'recall':    f"{a.get('recall', 0):.4f}",
            'f1':        f"{a.get('f1', 0):.4f}",
            'auc_roc':   f"{a.get('roc_auc', 0):.4f}",
            'avg_precision': f"{a.get('avg_precision', 0):.4f}",
            'opt_thresh':    f"{a.get('optimal_threshold', 0.5):.3f}",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'comparison_report.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON REPORT")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")

    # Print optimal thresholds with precision/recall breakdown
    print(f"\n  Optimal Threshold Tuning:")
    for tag in [t for _, _, t, _ in STEPS]:
        metrics_path = os.path.join(RESULTS_DIR, tag, 'combined', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                report = json.load(f)
            ot = report.get('combined', {}).get('optimal_threshold', 0.5)
            print(f"  {tag:20s}  optimal threshold: {ot:.3f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Phase 3 Production Pipeline')
    parser.add_argument('--start-step', type=int, default=1, help='Start from step (1-5)')
    parser.add_argument('--dry-run', action='store_true', help='Print steps without running')
    args = parser.parse_args()

    print(f"  Phase 3 Production Pipeline — {len(STEPS)} models + comparison")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Device:  CUDA enabled" if os.path.exists(VENV_PY) else "  Device:  CPU")

    for step_num, step_name, model_tag, timeout in STEPS:
        if step_num < args.start_step:
            print(f"\n  [SKIP] Step {step_num}: {step_name}")
            continue
        if args.dry_run:
            print(f"\n  [DRY-RUN] Step {step_num}: python train.py --dataset combined --model_type {model_tag}")
            continue
        code = run_step(step_num, step_name, model_tag, timeout)
        if code != 0:
            print(f"\n  Pipeline ABORTED at step: {step_name}")
            return code

    if not args.dry_run:
        build_comparison()

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  All results: {RESULTS_DIR}")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
