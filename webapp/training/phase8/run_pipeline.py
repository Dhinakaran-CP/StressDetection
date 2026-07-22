"""
run_pipeline.py — Sequential 4-model production pipeline.

Runs all 4 models on combined dataset with 5-fold LOSO CV.
Each model produces: ROC, PR-ROC, confusion matrix, classification report,
per-fold metrics CSV, aggregate JSON, predictions CSV, and checkpoint.

Usage:
    .\venv\Scripts\python webapp/training/phase8/run_pipeline.py
    .\venv\Scripts\python webapp/training/phase8/run_pipeline.py --dry-run
    .\venv\Scripts\python webapp/training/phase8/run_pipeline.py --start-step 3
"""
import os, sys, time, subprocess, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PIPELINE_DIR = os.path.join(ROOT, 'webapp', 'training', 'phase8')
VENV_PYTHON = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
REPORTS_BASE = os.path.join(ROOT, 'research', 'Phase_3_Production',
                            'production_model', 'ssvb_casa_ais_production')


MODELS = [
    ("SSVB-CASA-AIS (500K params, no GRL)",      'ssvb',             ['--dataset', 'combined', '--model_type', 'ssvb'],             90),
    ("CNNBaseline (21K params, no GRL)",           'cnn_baseline',    ['--dataset', 'combined', '--model_type', 'cnn_baseline'],      90),
    ("CNNBaseline+GRL (22K params, subj-GRL)",     'cnn_baseline_grl',['--dataset', 'combined', '--model_type', 'cnn_baseline_grl'],  90),
    ("ConvMoE-MF (8.8K params, dual GRL)",         'conv_moe_mf',     ['--dataset', 'combined', '--model_type', 'conv_moe_mf'],       90),
]


def run_step(step_name, script_path, args_list, timeout_min=10):
    cmd = [VENV_PYTHON, script_path] + args_list
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=timeout_min*60)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  [OK] {step_name} completed in {elapsed:.1f}s")
    else:
        print(f"\n  [FAIL] {step_name} exited with code {result.returncode}")
    return result.returncode


def build_comparison_report():
    """Aggregate metrics from all 4 models into a single comparison JSON + table."""
    import json, pandas as pd
    comparison = {}
    for _, model_tag, _, _ in MODELS:
        metrics_path = os.path.join(REPORTS_BASE, model_tag, 'combined', 'aggregate_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                comparison[model_tag] = json.load(f)
        else:
            comparison[model_tag] = None

    rows = []
    for model_tag, data in comparison.items():
        if data is None:
            rows.append({'model': model_tag, 'accuracy': 'N/A', 'f1': 'N/A',
                         'auc_roc': 'N/A', 'avg_precision': 'N/A'})
            continue
        agg = data.get('aggregate', {})
        rows.append({
            'model': model_tag,
            'accuracy': f"{agg.get('accuracy', 0):.4f}",
            'precision': f"{agg.get('precision', 0):.4f}",
            'recall': f"{agg.get('recall', 0):.4f}",
            'f1': f"{agg.get('f1', 0):.4f}",
            'auc_roc': f"{agg.get('roc_auc', 0):.4f}",
            'avg_precision': f"{agg.get('avg_precision', 'N/A')}",
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(REPORTS_BASE, 'comparison_report.csv')
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(REPORTS_BASE, 'comparison_report.json')
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON REPORT")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Full 4-model production pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Print steps without running')
    parser.add_argument('--start-step', type=int, default=1, help='Start from step N (1..6)')
    args = parser.parse_args()

    all_steps = []

    # Steps 1-4: Each model
    for idx, (step_name, model_tag, args_list, timeout) in enumerate(MODELS, 1):
        all_steps.append((
            f"WS{idx}: {step_name}",
            os.path.join(PIPELINE_DIR, 'train_ssvb_production.py'),
            args_list,
            timeout,
            model_tag,
        ))

    # Step 5: Comparison report
    all_steps.append((
        "WS5: Build Model Comparison Report",
        None,  # in-process
        "",
        2,
        None,
    ))

    print(f"\n{'='*60}")
    print(f"  FULL PRODUCTION PIPELINE — {len(all_steps)} stages")
    print(f"{'='*60}")
    for i, (name, _, _, _, _) in enumerate(all_steps, 1):
        print(f"  Step {i}: {name}")

    for idx, (step_name, script, args_list, timeout, model_tag) in enumerate(all_steps, 1):
        if idx < args.start_step:
            print(f"\n  [SKIP] Step {idx}: {step_name}")
            continue
        if args.dry_run:
            print(f"\n  [DRY-RUN] Step {idx}: python train_ssvb_production.py {' '.join(args_list)}")
            continue
        if script is None:
            # In-process step (comparison report)
            code = build_comparison_report()
        else:
            code = run_step(step_name, script, args_list, timeout)
        if code != 0:
            print(f"\n  Pipeline ABORTED at step: {step_name}")
            return code

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  All results: {REPORTS_BASE}")
    print(f"  Comparison: {os.path.join(REPORTS_BASE, 'comparison_report.csv')}")
    print(f"{'='*60}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
