"""
Master Research Pipeline Orchestrator
Runs all research model phases sequentially on clean data (StressID+WESAD only).

Usage:
    python scripts/run_research_pipeline.py                     # full pipeline
    python scripts/run_research_pipeline.py --phase phase3      # single phase
    python scripts/run_research_pipeline.py --skip phase3       # skip a phase
    python scripts/run_research_pipeline.py --exclude-dataset empathicschool
"""
import os, sys, time, json, subprocess, argparse, warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Detect python executable (prefer venv for GPU access)
PYTHON = None
for p in [os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe"),
          os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
          sys.executable]:
    if p and os.path.exists(p):
        PYTHON = p
        break
if not PYTHON:
    PYTHON = "python"

ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
CSV_DIR = os.path.join(PROJECT_ROOT, 'research', 'Phase_1_Baseline_LOSO')


# =====================================================================
# Phase Definitions
# =====================================================================
PHASES = [
    {
        'name': 'phase3_production',
        'label': 'Phase 3 Production Models (4 architectures)',
        'script': os.path.join('phase3_production', 'train.py'),
        'args': ['--exclude-dataset', 'empathicschool'],
        'data_check': lambda: os.path.exists(os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')),
        'env': {},
        'timeout': None,  # no timeout
    },
    {
        'name': 'phase1_baseline',
        'label': 'Phase 1 Baseline + Phase 2 Deep Models (18+ sklearn + PyTorch archs)',
        'script': os.path.join('scripts', 'train_and_evaluate_all.py'),
        'args': ['--mode', 'full_loso'],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase2_high_capacity',
        'label': 'Phase 2 High-Capacity Fusion Models (6 fusion architectures)',
        'script': os.path.join('research', 'Phase_2_High_Capacity', 'train_and_evaluate.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase4_temporal',
        'label': 'Phase 4 Temporal Deep Models (CNN-LSTM, GRU, LSTM, TCN, Transformer)',
        'script': os.path.join('scripts', 'run_temporal_pipeline.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase5_gan',
        'label': 'Phase 5 GAN-Augmented Models (7 temporal + GAN generator/critic)',
        'script': os.path.join('research', 'Phase_5_GAN_Augmentation', 'run_gan_pipeline.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase6_expert_gating',
        'label': 'Phase 6 Expert Gating Pipeline (SubpartExpert + GatingRouter)',
        'script': os.path.join('research', 'Phase_6_Expert_Gating', 'run_expert_pipeline.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase7_rf_specialist',
        'label': 'Phase 7 RF Specialist (Random Forest master pipeline)',
        'script': os.path.join('research', 'Phase_7_RF_Specialist', 'run_rf_master_pipeline.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(CSV_DIR, 'stress_features_fusion_5s.csv')),
        'env': {},
        'timeout': None,
    },
    {
        'name': 'phase8_production',
        'label': 'Phase 8 Production Training (Phase 8 production pipeline)',
        'script': os.path.join('webapp', 'training', 'phase8', 'train_ssvb_production.py'),
        'args': [],
        'data_check': lambda: os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'processed', 'certified_data')),
        'env': {},
        'timeout': None,
    },
]


# =====================================================================
# Helpers
# =====================================================================
def check_gpu():
    try:
        import subprocess
        r = subprocess.run([PYTHON, '-c', 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")'],
                          capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "check failed"
    except Exception as e:
        return f"error: {e}"


def run_phase(phase, skip_list, only_list):
    name = phase['name']
    if only_list and name not in only_list:
        return {'name': name, 'status': 'skipped (not in --only)'}
    if skip_list and name in skip_list:
        return {'name': name, 'status': 'skipped (--skip)'}
    if not phase['data_check']():
        return {'name': name, 'status': 'skipped (data not available)'}

    print(f"\n{'='*70}")
    print(f"  [{name}] {phase['label']}")
    print(f"  Script: {phase['script']}")
    print(f"{'='*70}")

    # Build env
    env = os.environ.copy()
    env.update(phase.get('env', {}))

    # Build command
    cmd = [PYTHON, phase['script']] + phase['args']

    start = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, cwd=PROJECT_ROOT,
            capture_output=True, text=True,
            timeout=phase.get('timeout')
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  [{name}] completed in {elapsed:.1f}s (exit 0)")
            status = 'success'
        else:
            # Print last 20 lines of stderr
            stderr_lines = result.stderr.strip().split('\n')
            print(f"  [{name}] FAILED (exit {result.returncode}) in {elapsed:.1f}s")
            print(f"  Last 20 stderr lines:")
            for line in stderr_lines[-20:]:
                print(f"    {line}")
            status = 'failed'
        return {'name': name, 'status': status, 'elapsed': elapsed, 'returncode': result.returncode}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  [{name}] TIMEOUT after {elapsed:.1f}s")
        return {'name': name, 'status': 'timeout', 'elapsed': elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [{name}] EXCEPTION: {e}")
        return {'name': name, 'status': 'error', 'elapsed': elapsed, 'error': str(e)}


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Master Research Pipeline Orchestrator')
    parser.add_argument('--phase', '--only', type=str, default=None,
                        help='Run only specific phase (comma-separated)')
    parser.add_argument('--skip', type=str, default=None,
                        help='Skip specific phases (comma-separated)')
    parser.add_argument('--exclude-dataset', '--exclude_dataset', type=str, default='empathicschool',
                        help='Dataset to exclude from training (default: empathicschool)')
    parser.add_argument('--list', action='store_true', help='List phases and exit')
    args = parser.parse_args()

    # Parse phase filters
    only_list = [p.strip() for p in args.phase.split(',')] if args.phase else None
    skip_list = [p.strip() for p in args.skip.split(',')] if args.skip else None

    if args.list:
        print(f"\n{'Phase':<25} {'Label'}")
        print('-' * 70)
        for p in PHASES:
            ok = 'yes' if p['data_check']() else 'NO'
            print(f'  {p["name"]:<25} {p["label"]}  [{ok}]')
        print(f'\nGPU: {check_gpu()}')
        print(f'Python: {PYTHON}')
        return

    print(f"{'='*70}")
    print(f"  Master Research Pipeline Orchestrator")
    print(f"  GPU: {check_gpu()}")
    print(f"  Python: {PYTHON}")
    print(f"  Excluding dataset: {args.exclude_dataset}")
    print(f"{'='*70}")

    results = []
    for phase in PHASES:
        results.append(run_phase(phase, skip_list, only_list))

    # Summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*70}")
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] in ('failed', 'timeout', 'error'))
    skipped = sum(1 for r in results if r['status'].startswith('skipped'))
    print(f"  Total: {len(results)}  Success: {success}  Failed: {failed}  Skipped: {skipped}")
    print()
    for r in results:
        icon = 'yes' if r['status'] == 'success' else ('x' if r['status'] in ('failed', 'timeout', 'error') else '-')
        elapsed = f" ({r.get('elapsed', 0):.0f}s)" if 'elapsed' in r else ''
        print(f"  {icon} {r['name']:<25} {r['status']}{elapsed}")
    print()


if __name__ == '__main__':
    main()
