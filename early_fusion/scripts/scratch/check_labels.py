import os
import csv

face_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\face_certified.csv"
voice_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\voice_certified.csv"
physio_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\physio_certified.csv"

def read_data(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            sub = row[0]
            task = row[1].lower()
            win_idx = row[3]
            win_start = float(row[4])
            win_end = float(row[5])
            label = int(row[6])
            
            key = (sub, task, win_idx)
            if key not in data:
                data[key] = []
            data[key].append((win_start, win_end, label))
    return data

face_data = read_data(face_path)
voice_data = read_data(voice_path)
physio_data = read_data(physio_path)

all_three_keys = set(face_data.keys()) & set(voice_data.keys()) & set(physio_data.keys())
print(f"Overlap all three keys: {len(all_three_keys)}")

exact_timing_matches = 0
label_mismatches = 0
label_matches = 0

for key in all_three_keys:
    f_start, f_end, f_label = face_data[key][0]
    v_start, v_end, v_label = voice_data[key][0]
    p_start, p_end, p_label = physio_data[key][0]
    
    if f_start == v_start == p_start and f_end == v_end == p_end:
        exact_timing_matches += 1
        
    if f_label == v_label == p_label:
        label_matches += 1
    else:
        label_mismatches += 1

print(f"Exact timing matches (diff == 0.0): {exact_timing_matches}")
print(f"Label matches: {label_matches}")
print(f"Label mismatches: {label_mismatches}")

# Let's also check label match for Face & Physio overlap
face_physio_keys = set(face_data.keys()) & set(physio_data.keys())
fp_label_matches = 0
fp_label_mismatches = 0
for key in face_physio_keys:
    f_start, f_end, f_label = face_data[key][0]
    p_start, p_end, p_label = physio_data[key][0]
    if f_label == p_label:
        fp_label_matches += 1
    else:
        fp_label_mismatches += 1
print(f"Face-Physio label matches: {fp_label_matches}, mismatches: {fp_label_mismatches}")

# Voice & Physio overlap
voice_physio_keys = set(voice_data.keys()) & set(physio_data.keys())
vp_label_matches = 0
vp_label_mismatches = 0
for key in voice_physio_keys:
    v_start, v_end, v_label = voice_data[key][0]
    p_start, p_end, p_label = physio_data[key][0]
    if v_label == p_label:
        vp_label_matches += 1
    else:
        vp_label_mismatches += 1
print(f"Voice-Physio label matches: {vp_label_matches}, mismatches: {vp_label_mismatches}")
