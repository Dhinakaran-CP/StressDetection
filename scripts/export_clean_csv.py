"""
Export clean CSV from enriched_training_data for legacy research models.
Filters to StressID+WESAD only (excludes EmpathicSchool and single-class subjects).
"""
import os, sys, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICHED_DIR = os.path.join(PROJECT_ROOT, 'data', 'enriched_training_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'research', 'Phase_1_Baseline_LOSO')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_clean_subjects(exclude_dataset='empathicschool'):
    """Return set of subject IDs to include (multi-class, not from excluded dataset)."""
    meta_path = os.path.join(ENRICHED_DIR, 'combined', 'metadata.parquet')
    if not os.path.exists(meta_path):
        print(f"  ERROR: enriched combined data not found at {meta_path}")
        return None
    meta = pd.read_parquet(meta_path)
    if exclude_dataset and 'dataset' in meta.columns:
        meta = meta[meta['dataset'] != exclude_dataset]
    # Identify single-class subjects
    subj_label_set = meta.groupby('subject_id')['label'].unique()
    single_class = set(subj_label_set[subj_label_set.apply(lambda x: len(x) < 2)].index)
    all_subjects = set(meta['subject_id'].unique())
    multi_class = all_subjects - single_class
    print(f"  Clean subjects: {len(multi_class)} multi-class (excluded {len(single_class)} single-class)")
    return multi_class


def filter_csv(source_path, output_path, clean_subjects):
    """Filter a CSV to only include rows from clean_subjects."""
    df = pd.read_csv(source_path)
    before = len(df)
    df = df[df['subject_id'].isin(clean_subjects)].reset_index(drop=True)
    after = len(df)
    print(f"  {os.path.basename(source_path)}: {before} -> {after} rows ({before - after} removed)")
    df.to_csv(output_path, index=False)
    return df


def main():
    print("=" * 60)
    print("  Export Clean CSV for Legacy Research Models")
    print("=" * 60)

    clean_subjects = get_clean_subjects()
    if clean_subjects is None:
        sys.exit(1)

    # Check for source CSVs — they might be at project root or various locations
    search_paths = [
        PROJECT_ROOT,
        os.path.join(PROJECT_ROOT, "scripts"),
        os.path.join(PROJECT_ROOT, "research", "Phase_1_Baseline_LOSO"),
    ]

    for scale in ['2sec', '5sec', '10sec']:
        filename = f'stress_features_fusion_{scale.replace("sec", "s")}'
        found = False
        for sp in search_paths:
            fp = os.path.join(sp, filename + '.csv')
            if os.path.exists(fp):
                out_path = os.path.join(OUTPUT_DIR, filename + '.csv')
                filter_csv(fp, out_path, clean_subjects)
                # Also place a copy in data_links
                link_dir = os.path.join(OUTPUT_DIR, 'data_links')
                os.makedirs(link_dir, exist_ok=True)
                link_path = os.path.join(link_dir, f'{scale}_fusion_features.csv')
                with open(link_path, 'w') as f:
                    f.write(f"Linked Source: {out_path}\n")
                found = True
                break
        if not found:
            print(f"  WARNING: {filename}.csv not found — skipping")

    print(f"\n  Clean CSVs exported to {OUTPUT_DIR}")
    print("  Run research models with: scripts/run_research_pipeline.py")


if __name__ == '__main__':
    main()
