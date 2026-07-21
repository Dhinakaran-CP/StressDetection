"""
Quick test of raw data loading.
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'webapp')
from webapp.training.phase8.clean_data_pipeline import *

print('=== StressID ===')
recs = load_stressid_raw()
print(f'Records: {len(recs)}')
subjs = set(r['subject'] for r in recs)
print(f'Unique subjects: {len(subjs)}')
sample = recs[0]
print(f'Sample: subj={sample["subject"]} task={sample["task"]} label={sample["label"]}')
video_exists = os.path.exists(sample['video_path']) if sample['video_path'] else False
audio_exists = os.path.exists(sample['audio_path']) if sample['audio_path'] else False
physio_exists = os.path.exists(sample['physio_path']) if sample['physio_path'] else False
print(f'  video={sample["video_path"]} exists={video_exists}')
print(f'  audio={sample["audio_path"]} exists={audio_exists}')
print(f'  physio={sample["physio_path"]} exists={physio_exists}')

print()
print('=== WESAD ===')
recs = load_wesad_raw()
print(f'Records: {len(recs)}')
for r in recs:
    pkl_ok = os.path.exists(r['pkl_path'])
    sz = os.path.getsize(r['pkl_path']) / 1e6 if pkl_ok else 0
    print(f'  {r["subject"]}: pkl={pkl_ok} size={sz:.0f}MB')

print()
print('=== EmpathicSchool ===')
recs = load_empathicschool_raw()
print(f'Records: {len(recs)}')
for r in recs:
    print(f'  {r["subject"]}: video={r["video_dir"] is not None} e4={r["e4_dir"] is not None} tags={r["tags_path"] is not None}')
