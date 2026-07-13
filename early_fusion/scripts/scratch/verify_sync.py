import os
import csv
import sys
from collections import Counter

# Paths to the certified CSV files
face_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\face_certified.csv"
voice_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\voice_certified.csv"
physio_path = r"e:\Document\GitHub\StressDetectionUsingML\certified_data\physio_certified.csv"

def analyze_csv(file_path, name):
    print(f"=== Analyzing {name} ===")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"Columns: {headers}")
        
        row_count = 0
        keys = {}
        subjects = set()
        tasks = set()
        
        for row in reader:
            row_count += 1
            if len(row) < 6:
                continue
            sub = row[0]
            task = row[1].lower() # Normalize to lowercase
            vid = row[2]
            win_idx = row[3]
            win_start = float(row[4])
            win_end = float(row[5])
            
            subjects.add(sub)
            tasks.add(row[1]) # Keep original casing for reporting
            
            key = (sub, task, win_idx)
            if key in keys:
                keys[key].append((win_start, win_end, row_count))
            else:
                keys[key] = [(win_start, win_end, row_count)]
                
        print(f"Total Rows: {row_count}")
        print(f"Unique Subjects: {len(subjects)}")
        print(f"Unique Tasks: {len(tasks)}")
        print(f"Unique (subject_id, normalized_task_id, window_index) Keys: {len(keys)}")
        
        sizes = []
        task_windows = {}
        for key, vals in keys.items():
            sub, task, win_idx = key
            for win_start, win_end, rc in vals:
                sizes.append(win_end - win_start)
                tk = (sub, task)
                if tk not in task_windows:
                    task_windows[tk] = []
                task_windows[tk].append((int(win_idx), win_start, win_end))
                
        size_counts = Counter(sizes)
        print(f"Window Sizes (duration = end - start): {dict(size_counts)}")
        
        stride_counts = Counter()
        for tk, wins in task_windows.items():
            wins.sort()
            for i in range(len(wins) - 1):
                curr_idx, curr_start, curr_end = wins[i]
                nxt_idx, nxt_start, nxt_end = wins[i+1]
                if nxt_idx == curr_idx + 1:
                    stride_counts[nxt_start - curr_start] += 1
                    
        print(f"Strides (start of next window - start of current): {dict(stride_counts)}")
        print()
        
        return {
            'keys': keys,
            'subjects': subjects,
            'tasks': tasks,
            'row_count': row_count,
            'window_sizes': size_counts,
            'strides': stride_counts
        }

print("Starting normalized analysis...")
face_data = analyze_csv(face_path, "Face")
voice_data = analyze_csv(voice_path, "Voice")
physio_data = analyze_csv(physio_path, "Physio")

if face_data and voice_data and physio_data:
    print("=== Cross-Modality Comparison (Normalized Tasks) ===")
    
    all_keys = set(face_data['keys'].keys()) | set(voice_data['keys'].keys()) | set(physio_data['keys'].keys())
    print(f"Total Union of Keys: {len(all_keys)}")
    
    duplicate_face = 0
    duplicate_voice = 0
    duplicate_physio = 0
    
    timing_mismatches = 0
    timing_matches = 0
    
    modality_coverage = {
        'face_only': 0,
        'voice_only': 0,
        'physio_only': 0,
        'face_voice': 0,
        'face_physio': 0,
        'voice_physio': 0,
        'all_three': 0
    }
    
    mismatched_sample_keys = []
    
    for key in all_keys:
        has_face = key in face_data['keys']
        has_voice = key in voice_data['keys']
        has_physio = key in physio_data['keys']
        
        if has_face and has_voice and has_physio:
            modality_coverage['all_three'] += 1
        elif has_face and has_voice:
            modality_coverage['face_voice'] += 1
        elif has_face and has_physio:
            modality_coverage['face_physio'] += 1
        elif has_voice and has_physio:
            modality_coverage['voice_physio'] += 1
        elif has_face:
            modality_coverage['face_only'] += 1
        elif has_voice:
            modality_coverage['voice_only'] += 1
        elif has_physio:
            modality_coverage['physio_only'] += 1
            
        if has_face and len(face_data['keys'][key]) > 1:
            duplicate_face += len(face_data['keys'][key]) - 1
        if has_voice and len(voice_data['keys'][key]) > 1:
            duplicate_voice += len(voice_data['keys'][key]) - 1
        if has_physio and len(physio_data['keys'][key]) > 1:
            duplicate_physio += len(physio_data['keys'][key]) - 1
            
        # Check temporal agreement
        if has_face and has_voice and has_physio:
            f_times = face_data['keys'][key]
            v_times = voice_data['keys'][key]
            p_times = physio_data['keys'][key]
            
            f_start, f_end, _ = f_times[0]
            v_start, v_end, _ = v_times[0]
            p_start, p_end, _ = p_times[0]
            
            tolerance = 0.05
            start_ok = abs(f_start - v_start) <= tolerance and abs(f_start - p_start) <= tolerance and abs(v_start - p_start) <= tolerance
            end_ok = abs(f_end - v_end) <= tolerance and abs(f_end - p_end) <= tolerance and abs(v_end - p_end) <= tolerance
            
            if start_ok and end_ok:
                timing_matches += 1
            else:
                timing_mismatches += 1
                if len(mismatched_sample_keys) < 10:
                    mismatched_sample_keys.append((key, f_start, f_end, v_start, v_end, p_start, p_end))
                
    print(f"Modality Coverage:")
    for k, v in modality_coverage.items():
        print(f"  {k}: {v}")
        
    print(f"\nDuplicates:")
    print(f"  Face duplicates: {duplicate_face}")
    print(f"  Voice duplicates: {duplicate_voice}")
    print(f"  Physio duplicates: {duplicate_physio}")
    
    print(f"\nTiming check (for samples with all three modalities):")
    print(f"  Matched timing: {timing_matches}")
    print(f"  Mismatched timing: {timing_mismatches}")
    
    if mismatched_sample_keys:
        print("\nSome mismatched timing examples (key, face_start/end, voice_start/end, physio_start/end):")
        for ex in mismatched_sample_keys:
            print(f"  Key: {ex[0]}")
            print(f"    Face:  {ex[1]:.3f} - {ex[2]:.3f}")
            print(f"    Voice: {ex[3]:.3f} - {ex[4]:.3f}")
            print(f"    Physio:{ex[5]:.3f} - {ex[6]:.3f}")
            
    # Let's check overlap of (subject, task) pairs for Face, Voice, Physio
    face_subj_tasks = set((sub, task) for sub, task, win in face_data['keys'].keys())
    voice_subj_tasks = set((sub, task) for sub, task, win in voice_data['keys'].keys())
    physio_subj_tasks = set((sub, task) for sub, task, win in physio_data['keys'].keys())
    
    print(f"\nSubject-Task Overlaps:")
    print(f"  Face (subject, task) pairs: {len(face_subj_tasks)}")
    print(f"  Voice (subject, task) pairs: {len(voice_subj_tasks)}")
    print(f"  Physio (subject, task) pairs: {len(physio_subj_tasks)}")
    print(f"  Union of all (subject, task) pairs: {len(face_subj_tasks | voice_subj_tasks | physio_subj_tasks)}")
    print(f"  Intersection of all three: {len(face_subj_tasks & voice_subj_tasks & physio_subj_tasks)}")
    print(f"  Intersection of Face & Physio: {len(face_subj_tasks & physio_subj_tasks)}")
    
    # Are there subjects present in Voice but not in Face?
    print(f"\nVoice subjects not in Face: {voice_data['subjects'] - face_data['subjects']}")
    print(f"Face subjects not in Voice: {face_data['subjects'] - voice_data['subjects']}")
    print(f"Physio subjects not in Face: {physio_data['subjects'] - face_data['subjects']}")
    print(f"Physio subjects not in Voice: {physio_data['subjects'] - voice_data['subjects']}")
