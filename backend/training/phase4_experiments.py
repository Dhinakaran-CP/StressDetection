import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

def run_loso_cv(X, y, groups, model, scaler=None):
    gkf = GroupKFold(n_splits=3)
    accs, f1s = [], []
    oof_preds = np.zeros(len(y))
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if scaler:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            oof_preds[test_idx] = model.predict_proba(X_test)[:, 1]
        else:
            oof_preds[test_idx] = preds
            
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        
    return np.mean(accs), np.mean(f1s), oof_preds

def subject_aware_normalization(df, feature_cols):
    df_norm = df.copy()
    for subject in df['subject_id'].unique():
        sub_mask = df['subject_id'] == subject
        calm_mask = sub_mask & (df['label'] == 0)
        
        if calm_mask.sum() == 0:
            mean_vals = df.loc[sub_mask, feature_cols].mean()
        else:
            mean_vals = df.loc[calm_mask, feature_cols].mean()
            
        df_norm.loc[sub_mask, feature_cols] = df.loc[sub_mask, feature_cols] - mean_vals
    return df_norm

def apply_temporal_windowing(df, feature_cols, window_size=2):
    """Averages features over consecutive windows within the same task."""
    df_grouped = df.copy()
    # Sort just to be safe
    df_grouped = df_grouped.sort_values(by=['subject_id', 'task_id', 'window_index'])
    # Rolling mean on features, grouped by subject and task
    df_grouped[feature_cols] = df_grouped.groupby(['subject_id', 'task_id'])[feature_cols].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
    return df_grouped

def run_modality_experiments(modality_name, df_path, feature_cols, base_model):
    print(f"\n{'='*50}\nStarting Experiments for {modality_name.upper()}\n{'='*50}")
    t0 = time.time()
    df = pd.read_csv(df_path)
    df = df.dropna(subset=feature_cols)
    
    # Downsample voice/face if too large to make experiments fast (optional, but let's just rely on n_jobs=-1 and fast RF)
    y = df['label'].values
    groups = df['subject_id'].values
    X_raw = df[feature_cols].values
    
    results = []
    
    print("  -> 1. Running Baseline...")
    acc, f1, oof_base = run_loso_cv(X_raw, y, groups, base_model, scaler=True)
    results.append({"Experiment": "1. Baseline (Raw + Scaler)", "Accuracy": acc, "F1": f1})
    
    print("  -> 2. Running Subject-Aware Normalization...")
    df_norm = subject_aware_normalization(df, feature_cols)
    X_norm = df_norm[feature_cols].values
    acc, f1, oof_norm = run_loso_cv(X_norm, y, groups, base_model, scaler=True)
    results.append({"Experiment": "2. Subject-Aware Normalization", "Accuracy": acc, "F1": f1})
    
    print("  -> 3. Running Temporal Windowing (2.0s context)...")
    df_win = apply_temporal_windowing(df_norm, feature_cols, window_size=2)
    X_win = df_win[feature_cols].values
    acc, f1, oof_win = run_loso_cv(X_win, y, groups, base_model, scaler=True)
    results.append({"Experiment": "3. Temporal Windowing (Norm + Win)", "Accuracy": acc, "F1": f1})
    
    print("  -> 4. Running Feature Selection (Top 50%)...")
    k = max(1, len(feature_cols) // 2)
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X_win, y)
    acc, f1, _ = run_loso_cv(X_sel, y, groups, base_model, scaler=True)
    results.append({"Experiment": f"4. Feature Selection (Top {k})", "Accuracy": acc, "F1": f1})
    
    print("  -> 5. Running Heavy RF Model...")
    heavy_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1)
    acc, f1, _ = run_loso_cv(X_win, y, groups, heavy_model, scaler=True)
    results.append({"Experiment": "5. Heavier Tuned Model on Win", "Accuracy": acc, "F1": f1})
    
    # Pick the best OOF dynamically based on accuracy (between base, norm, win)
    best_acc = max(results[0]["Accuracy"], results[1]["Accuracy"], results[2]["Accuracy"])
    best_oof = oof_base
    if results[1]["Accuracy"] == best_acc: best_oof = oof_norm
    if results[2]["Accuracy"] == best_acc: best_oof = oof_win
    
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    print(f"Time taken: {time.time()-t0:.1f}s")
    
    return df, best_oof

def run_fusion_experiments(face_df, voice_df, physio_df, face_oof, voice_oof, physio_oof):
    print(f"\n{'='*50}\nStarting Experiments for FUSION\n{'='*50}")
    face_df['face_oof'] = face_oof
    voice_df['voice_oof'] = voice_oof
    physio_df['physio_oof'] = physio_oof
    
    merged = pd.merge(
        face_df[['subject_id', 'window_index', 'label', 'face_oof']],
        voice_df[['subject_id', 'window_index', 'label', 'voice_oof']],
        on=['subject_id', 'window_index', 'label'],
        how='inner'
    )
    merged = pd.merge(
        merged,
        physio_df[['subject_id', 'window_index', 'label', 'physio_oof']],
        on=['subject_id', 'window_index', 'label'],
        how='inner'
    )
    
    if len(merged) == 0:
        print("Failed to align datasets for fusion.")
        return
        
    print(f"Aligned {len(merged)} windows for fusion testing.")
    y = merged['label'].values
    groups = merged['subject_id'].values
    
    results = []
    
    # 1. Naive Average (equal weights)
    preds_avg = (merged['face_oof'] + merged['voice_oof'] + merged['physio_oof']) / 3.0
    acc_avg = accuracy_score(y, (preds_avg > 0.5).astype(int))
    f1_avg = f1_score(y, (preds_avg > 0.5).astype(int), zero_division=0)
    results.append({"Experiment": "1. Naive Average Fusion (3-way)", "Accuracy": acc_avg, "F1": f1_avg})
    
    # 2. Weighted Average Grid Search
    best_w = (0.33, 0.33, 0.33)
    best_acc = acc_avg
    best_f1 = f1_avg
    
    for wf in np.arange(0.1, 0.9, 0.1):
        for wv in np.arange(0.1, 0.9 - wf + 0.01, 0.1):
            wp = 1.0 - wf - wv
            if wp < 0.05: continue
            
            preds_w = wf * merged['face_oof'] + wv * merged['voice_oof'] + wp * merged['physio_oof']
            acc_w = accuracy_score(y, (preds_w > 0.5).astype(int))
            f1_w = f1_score(y, (preds_w > 0.5).astype(int), zero_division=0)
            
            if acc_w > best_acc:
                best_acc = acc_w
                best_f1 = f1_w
                best_w = (wf, wv, wp)
                
    results.append({
        "Experiment": f"2. Best Weighted Fusion (F:{best_w[0]:.2f}, V:{best_w[1]:.2f}, P:{best_w[2]:.2f})", 
        "Accuracy": best_acc, 
        "F1": best_f1
    })
    
    # 3. Logistic Stacking
    X_stack = merged[['face_oof', 'voice_oof', 'physio_oof']].values
    stacker = LogisticRegression(class_weight='balanced')
    acc_stack, f1_stack, _ = run_loso_cv(X_stack, y, groups, stacker, scaler=False)
    results.append({"Experiment": "3. Logistic Stacking (OOF, 3-way)", "Accuracy": acc_stack, "F1": f1_stack})
    
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    import yaml
    with open("contracts/feature_contract.yaml", "r") as f:
        contract = yaml.safe_load(f)
        
    face_cols = contract["modalities"]["face"]["features"]
    voice_cols = contract["modalities"]["voice"]["features"]
    physio_cols = contract["modalities"]["physio"]["features"]
    
    # Very fast models with parallelization
    face_base = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    voice_base = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    physio_base = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    
    face_df, face_oof = run_modality_experiments("Face", "dataset_certified/face_certified.csv", face_cols, face_base)
    voice_df, voice_oof = run_modality_experiments("Voice", "dataset_certified/voice_certified.csv", voice_cols, voice_base)
    physio_df, physio_oof = run_modality_experiments("Physio", "dataset_certified/physio_certified.csv", physio_cols, physio_base)
    
    run_fusion_experiments(face_df, voice_df, physio_df, face_oof, voice_oof, physio_oof)
