import pandas as pd
import json, os

root = 'data/enriched_training_data'
for ds in ['stressid', 'wesad', 'empathicschool', 'combined']:
    p = os.path.join(root, ds, 'metadata.parquet')
    d = os.path.join(root, ds, 'group_dims.json')
    if os.path.exists(p):
        m = pd.read_parquet(p)
        dims = json.load(open(d))
        subj = m['subject_id'].nunique()
        label_sum = m['label'].sum()
        label_mean = 100 * m['label'].mean()
        print(f'{ds:15s}: {len(m):6d} windows, {subj:3d} subjects, '
              f'stress={label_sum:5d} ({label_mean:.1f}%)')
        print(f'  Groups: {json.dumps(dims)}')
