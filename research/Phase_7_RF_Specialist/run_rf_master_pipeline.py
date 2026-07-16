import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import joblib

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Directory Setup
# ---------------------------------------------------------
backend_dir = r"c:\Users\StressProject\Desktop\StressDetectionUsingML"
loso_dir = os.path.join(backend_dir, "research", "Phase_1_Baseline_LOSO")

# Separate folder for this execution phase
RUN_DIR = os.path.join(backend_dir, "research", "Phase_7_RF_Specialist")
OUTPUTS_DIR = os.path.join(RUN_DIR, "outputs", "random_forest_master")
os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "reports"), exist_ok=True)

# Set random seed
np.random.seed(42)

# ---------------------------------------------------------
# Step 2: Helper Functions
# ---------------------------------------------------------
def evaluate_predictions(targets, preds, probs, elapsed_time):
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="binary", zero_division=0)
    rec = recall_score(targets, preds, average="binary", zero_division=0)
    f1 = f1_score(targets, preds, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(targets, preds)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.5
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Balanced-Accuracy": bal_acc,
        "ROC-AUC": auc,
        "Runtime-Seconds": elapsed_time
    }

def save_plots(targets, preds, probs, model_name, save_dir):
    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Calm', 'Stress'])
    plt.yticks(tick_marks, ['Calm', 'Stress'])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(targets, probs):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()
    
    # Precision-Recall Curve
    pr_y, pr_x, _ = precision_recall_curve(targets, probs)
    plt.figure()
    plt.plot(pr_x, pr_y, color='darkgreen', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"))
    plt.close()

# ---------------------------------------------------------
# Step 3: Core Pipeline Execution
# ---------------------------------------------------------
def run_rf_scale(scale, filename):
    start_time = time.time()
    print(f"\n==========================================================")
    print(f"  RUNNING TIMEFRAME SCALE: {scale} ({filename})")
    print(f"==========================================================\n")
    
    file_path = os.path.join(backend_dir, "data", "features", filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(loso_dir, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(backend_dir, filename)
            if not os.path.exists(file_path):
                print(f"[WARNING] Feature store {filename} not found. Skipping scale {scale}.")
                return []
            
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    subj_list = df["subject_id"].unique().tolist()
    labels = df["label"].astype(int).values
    subjects = df["subject_id"].values
    
    exclude_cols = ["subject_id", "task_id", "window_index", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_abs")]
    
    # Define Specialist feature masks
    face_feats = [c for c in feature_cols if c.startswith("face_")]
    voice_feats = [c for c in feature_cols if c.startswith("voice_")]
    physio_feats = [c for c in feature_cols if c.startswith("ecg_") or c.startswith("eda_") or c.startswith("resp_") or c.startswith("quality_") or c.startswith("physio_")]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Subpart splits - Face: {len(face_feats)}, Voice: {len(voice_feats)}, Physio: {len(physio_feats)}")
    
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(df, labels, groups=subjects))
    
    # ---------------------------------------------------------
    # Phase A: Single Forest Tuning on First Fold
    # ---------------------------------------------------------
    print("--> Hyperparameter tuning on first fold training split...")
    first_train_idx, _ = splits[0]
    X_tune = df[feature_cols].iloc[first_train_idx].fillna(0).values
    y_tune = labels[first_train_idx]
    
    scaler_tune = StandardScaler()
    X_tune_norm = scaler_tune.fit_transform(X_tune)
    
    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", "balanced_subsample", None],
        "bootstrap": [True]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        rf_base, param_distributions=param_dist, n_iter=5, cv=3,
        n_jobs=-1, scoring="f1", random_state=42
    )
    search.fit(X_tune_norm, y_tune)
    best_params = search.best_params_
    print(f"Best hyperparameters found: {best_params}")
    
    scale_dir = os.path.join(OUTPUTS_DIR, scale)
    os.makedirs(scale_dir, exist_ok=True)
    
    # Instantiating directories
    single_dir = os.path.join(scale_dir, "single_forest")
    spec_dir = os.path.join(scale_dir, "specialist_forests")
    comb_dir = os.path.join(scale_dir, "combined_ensemble")
    metrics_dir = os.path.join(scale_dir, "metrics")
    plots_dir = os.path.join(scale_dir, "plots")
    reports_dir = os.path.join(scale_dir, "reports")
    
    for d in [single_dir, spec_dir, comb_dir, metrics_dir, plots_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Accumulated outputs for evaluation
    targets_all = []
    
    # Single forest predictions
    single_preds_all, single_probs_all = [], []
    single_runtimes = []
    
    # Specialists predictions
    face_preds_all, face_probs_all, face_runtimes = [], [], []
    voice_preds_all, voice_probs_all, voice_runtimes = [], [], []
    physio_preds_all, physio_probs_all, physio_runtimes = [], [], []
    
    # Feature importance storage
    feature_importances_accum = np.zeros(len(feature_cols))
    
    # ---------------------------------------------------------
    # Phase B: Evaluation Loop over Folds
    # ---------------------------------------------------------
    fold_idx = 1
    for train_idx, val_idx in splits:
        targets_all.extend(labels[val_idx])
        
        # 1. Baseline Single Forest
        t_start = time.time()
        scaler_s = StandardScaler()
        X_tr_s = scaler_s.fit_transform(df[feature_cols].iloc[train_idx].fillna(0).values)
        X_val_s = scaler_s.transform(df[feature_cols].iloc[val_idx].fillna(0).values)
        
        clf_single = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        clf_single.fit(X_tr_s, labels[train_idx])
        
        single_preds_all.extend(clf_single.predict(X_val_s))
        single_probs_all.extend(clf_single.predict_proba(X_val_s)[:, 1])
        single_runtimes.append(time.time() - t_start)
        
        # Accumulate feature importances
        feature_importances_accum += clf_single.feature_importances_
        
        # 2. Face Specialist
        t_start = time.time()
        if len(face_feats) > 0:
            scaler_face = StandardScaler()
            X_tr_face = scaler_face.fit_transform(df[face_feats].iloc[train_idx].fillna(0).values)
            X_val_face = scaler_face.transform(df[face_feats].iloc[val_idx].fillna(0).values)
            
            clf_face = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            clf_face.fit(X_tr_face, labels[train_idx])
            
            face_preds_all.extend(clf_face.predict(X_val_face))
            face_probs_all.extend(clf_face.predict_proba(X_val_face)[:, 1])
            face_runtimes.append(time.time() - t_start)
        else:
            face_preds_all.extend(np.zeros(len(val_idx)))
            face_probs_all.extend(np.zeros(len(val_idx)))
            face_runtimes.append(0)
            
        # 3. Voice Specialist
        t_start = time.time()
        if len(voice_feats) > 0:
            scaler_voice = StandardScaler()
            X_tr_voice = scaler_voice.fit_transform(df[voice_feats].iloc[train_idx].fillna(0).values)
            X_val_voice = scaler_voice.transform(df[voice_feats].iloc[val_idx].fillna(0).values)
            
            clf_voice = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            clf_voice.fit(X_tr_voice, labels[train_idx])
            
            voice_preds_all.extend(clf_voice.predict(X_val_voice))
            voice_probs_all.extend(clf_voice.predict_proba(X_val_voice)[:, 1])
            voice_runtimes.append(time.time() - t_start)
        else:
            voice_preds_all.extend(np.zeros(len(val_idx)))
            voice_probs_all.extend(np.zeros(len(val_idx)))
            voice_runtimes.append(0)
            
        # 4. Physiology Specialist
        t_start = time.time()
        if len(physio_feats) > 0:
            scaler_physio = StandardScaler()
            X_tr_physio = scaler_physio.fit_transform(df[physio_feats].iloc[train_idx].fillna(0).values)
            X_val_physio = scaler_physio.transform(df[physio_feats].iloc[val_idx].fillna(0).values)
            
            clf_physio = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            clf_physio.fit(X_tr_physio, labels[train_idx])
            
            physio_preds_all.extend(clf_physio.predict(X_val_physio))
            physio_probs_all.extend(clf_physio.predict_proba(X_val_physio)[:, 1])
            physio_runtimes.append(time.time() - t_start)
        else:
            physio_preds_all.extend(np.zeros(len(val_idx)))
            physio_probs_all.extend(np.zeros(len(val_idx)))
            physio_runtimes.append(0)
            
        fold_idx += 1
        
    targets_all = np.array(targets_all)
    single_preds_all = np.array(single_preds_all)
    single_probs_all = np.array(single_probs_all)
    
    face_preds_all = np.array(face_preds_all)
    face_probs_all = np.array(face_probs_all)
    
    voice_preds_all = np.array(voice_preds_all)
    voice_probs_all = np.array(voice_probs_all)
    
    physio_preds_all = np.array(physio_preds_all)
    physio_probs_all = np.array(physio_probs_all)
    
    # 5. Combined Gating Ensemble (Soft Voting)
    fused_probs_all = (face_probs_all + voice_probs_all + physio_probs_all) / 3.0
    fused_preds_all = (fused_probs_all >= 0.5).astype(int)
    fused_runtime = np.sum(face_runtimes) + np.sum(voice_runtimes) + np.sum(physio_runtimes)
    elapsed_time = time.time() - start_time
    
    # ---------------------------------------------------------
    # Phase C: Save Artifacts and Reports
    # ---------------------------------------------------------
    # Evaluate metrics
    metrics_single = evaluate_predictions(targets_all, single_preds_all, single_probs_all, np.sum(single_runtimes))
    metrics_face = evaluate_predictions(targets_all, face_preds_all, face_probs_all, np.sum(face_runtimes))
    metrics_voice = evaluate_predictions(targets_all, voice_preds_all, voice_probs_all, np.sum(voice_runtimes))
    metrics_physio = evaluate_predictions(targets_all, physio_preds_all, physio_probs_all, np.sum(physio_runtimes))
    metrics_ensemble = evaluate_predictions(targets_all, fused_preds_all, fused_probs_all, fused_runtime)
    
    # Save checkpoints
    joblib.dump(clf_single, os.path.join(single_dir, "model.pkl"))
    if len(face_feats) > 0: joblib.dump(clf_face, os.path.join(spec_dir, "face_specialist.pkl"))
    if len(voice_feats) > 0: joblib.dump(clf_voice, os.path.join(spec_dir, "voice_specialist.pkl"))
    if len(physio_feats) > 0: joblib.dump(clf_physio, os.path.join(spec_dir, "physio_specialist.pkl"))
    
    # Save CSV reports
    pd.DataFrame(metrics_single, index=[0]).T.to_csv(os.path.join(single_dir, "metrics.csv"))
    pd.DataFrame(metrics_ensemble, index=[0]).T.to_csv(os.path.join(comb_dir, "metrics.csv"))
    
    pd.DataFrame({
        "Classifier": ["Single Forest", "Face Specialist", "Voice Specialist", "Physio Specialist", "Combined Ensemble"],
        "Accuracy": [metrics_single["Accuracy"], metrics_face["Accuracy"], metrics_voice["Accuracy"], metrics_physio["Accuracy"], metrics_ensemble["Accuracy"]],
        "Precision": [metrics_single["Precision"], metrics_face["Precision"], metrics_voice["Precision"], metrics_physio["Precision"], metrics_ensemble["Precision"]],
        "Recall": [metrics_single["Recall"], metrics_face["Recall"], metrics_voice["Recall"], metrics_physio["Recall"], metrics_ensemble["Recall"]],
        "F1-Score": [metrics_single["F1-Score"], metrics_face["F1-Score"], metrics_voice["F1-Score"], metrics_physio["F1-Score"], metrics_ensemble["F1-Score"]],
        "ROC-AUC": [metrics_single["ROC-AUC"], metrics_face["ROC-AUC"], metrics_voice["ROC-AUC"], metrics_physio["ROC-AUC"], metrics_ensemble["ROC-AUC"]],
        "Runtime": [metrics_single["Runtime-Seconds"], metrics_face["Runtime-Seconds"], metrics_voice["Runtime-Seconds"], metrics_physio["Runtime-Seconds"], metrics_ensemble["Runtime-Seconds"]]
    }).to_csv(os.path.join(metrics_dir, "metrics_comparison.csv"), index=False)
    
    # Save plots
    save_plots(targets_all, single_preds_all, single_probs_all, "Single Forest", plots_dir)
    save_plots(targets_all, fused_preds_all, fused_probs_all, "Combined Ensemble", plots_dir)
    
    # Plot feature importances for Single Forest
    avg_feat_importances = feature_importances_accum / 5.0
    top_indices = np.argsort(avg_feat_importances)[-15:]
    top_feats = [feature_cols[i] for i in top_indices]
    top_scores = avg_feat_importances[top_indices]
    
    plt.figure(figsize=(10, 5))
    plt.barh(top_feats, top_scores, color='purple', align='center')
    plt.xlabel('Average Feature Importance')
    plt.title('Top 15 Feature Importances (Tuned Single Forest)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
    plt.close()
    
    # Plot specialist comparison bar chart
    plt.figure(figsize=(10, 5))
    scores_comp = [metrics_single["F1-Score"], metrics_face["F1-Score"], metrics_voice["F1-Score"], metrics_physio["F1-Score"], metrics_ensemble["F1-Score"]]
    labels_comp = ["Tuned Single Forest", "Face Specialist", "Voice Specialist", "Physio Specialist", "Combined Ensemble"]
    colors = ['blue', 'gray', 'gray', 'gray', 'orange']
    plt.bar(labels_comp, scores_comp, color=colors, width=0.5)
    plt.ylabel('F1-Score')
    plt.title('Specialist and Ensemble Model Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "specialist_comparison.png"))
    plt.close()
    
    # Save configs
    with open(os.path.join(scale_dir, "config.json"), "w") as cfg_f:
        json.dump({
            "scale": scale,
            "tuned_hyperparameters": best_params,
            "feature_subsets": {
                "face": len(face_feats),
                "voice": len(voice_feats),
                "physiology": len(physio_feats)
            },
            "ensemble_voting": "soft",
            "seed": 42
        }, cfg_f, indent=4)
        
    # Decision rule verification
    ensemble_gain = metrics_ensemble["F1-Score"] - metrics_single["F1-Score"]
    ensemble_selected = ensemble_gain > 0.005
    decision = "PROMOTED Combined Ensemble (improved > 0.5% F1-score)" if ensemble_selected else "RETAINED Tuned Single Forest (ensemble gain too small or negative)"
    
    # Summary scorecard
    with open(os.path.join(reports_dir, "summary.md"), "w") as summary_f:
        summary_f.write(f"# Benchmarking Scorecard: Random Forest Master ({scale})\n\n")
        summary_f.write(f"This scorecard compares the tuned baseline champion Random Forest with the specialist modality classifiers and the combined soft-voting ensemble.\n\n")
        summary_f.write(f"### Performance Metrics Comparison\n")
        summary_f.write(f"| Model / Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Runtime (s) |\n")
        summary_f.write(f"| --- | --- | --- | --- | --- | --- | --- |\n")
        summary_f.write(f"| **Tuned Single Forest** | {metrics_single['Accuracy']:.4f} | {metrics_single['Precision']:.4f} | {metrics_single['Recall']:.4f} | {metrics_single['F1-Score']:.4f} | {metrics_single['ROC-AUC']:.4f} | {metrics_single['Runtime-Seconds']:.2f} |\n")
        summary_f.write(f"| Face Specialist | {metrics_face['Accuracy']:.4f} | {metrics_face['Precision']:.4f} | {metrics_face['Recall']:.4f} | {metrics_face['F1-Score']:.4f} | {metrics_face['ROC-AUC']:.4f} | {metrics_face['Runtime-Seconds']:.2f} |\n")
        summary_f.write(f"| Voice Specialist | {metrics_voice['Accuracy']:.4f} | {metrics_voice['Precision']:.4f} | {metrics_voice['Recall']:.4f} | {metrics_voice['F1-Score']:.4f} | {metrics_voice['ROC-AUC']:.4f} | {metrics_voice['Runtime-Seconds']:.2f} |\n")
        summary_f.write(f"| Physio Specialist | {metrics_physio['Accuracy']:.4f} | {metrics_physio['Precision']:.4f} | {metrics_physio['Recall']:.4f} | {metrics_physio['F1-Score']:.4f} | {metrics_physio['ROC-AUC']:.4f} | {metrics_physio['Runtime-Seconds']:.2f} |\n")
        summary_f.write(f"| **Combined Ensemble** | {metrics_ensemble['Accuracy']:.4f} | {metrics_ensemble['Precision']:.4f} | {metrics_ensemble['Recall']:.4f} | {metrics_ensemble['F1-Score']:.4f} | {metrics_ensemble['ROC-AUC']:.4f} | {metrics_ensemble['Runtime-Seconds']:.2f} |\n\n")
        summary_f.write(f"### Final Selection Decision\n")
        summary_f.write(f"* **Decision**: **{decision}** (Ensemble F1 Gain = {ensemble_gain:.4f})\n")
        
    return {
        "Scale": scale,
        "Tuned Single Forest Accuracy": metrics_single["Accuracy"],
        "Tuned Single Forest F1": metrics_single["F1-Score"],
        "Combined Ensemble Accuracy": metrics_ensemble["Accuracy"],
        "Combined Ensemble F1": metrics_ensemble["F1-Score"],
        "Selection Decision": decision,
        "Runtime": elapsed_time
    }

def main():
    parser = argparse.ArgumentParser(description="Random Forest Master Tuning & Specialist Ensemble Pipeline")
    parser.add_argument("--scale", type=str, default="all", choices=["2sec", "5sec", "10sec", "all"],
                        help="Timeframe scale to run (default: all)")
    args = parser.parse_args()
    
    scales_configs = [
        ("2sec", "stress_features_fusion_2s.csv"),
        ("5sec", "stress_features_fusion_5s.csv"),
        ("10sec", "stress_features_fusion_10s.csv")
    ]
    
    if args.scale != "all":
        scales_configs = [item for item in scales_configs if item[0] == args.scale]
        
    global_results = []
    
    for scale, filename in scales_configs:
        res = run_rf_scale(scale, filename)
        global_results.append(res)
        
    if len(global_results) > 0:
        df_global = pd.DataFrame(global_results)
        df_global.to_csv(os.path.join(RUN_DIR, "reports", "rf_master_leaderboard.csv"), index=False)
        
        # Build consolidated report
        report_path = os.path.join(RUN_DIR, "reports", "rf_master_comparison_report.md")
        with open(report_path, "w") as f:
            f.write("# Random Forest Master & Specialist Ensemble Comparison Report\n\n")
            f.write("This report compiles performance comparisons for the Tuned Single Random Forest and the Combined Specialist Ensemble across all window scales (2s, 5s, 10s).\n\n")
            
            headers = list(df_global.columns)
            md_table = "| " + " | ".join(headers) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for _, row in df_global.iterrows():
                vals = []
                for val in row:
                    if isinstance(val, float):
                        vals.append(f"{val:.4f}")
                    else:
                        vals.append(str(val))
                md_table += "| " + " | ".join(vals) + " |\n"
                
            f.write(md_table)
            f.write("\n\n*All plots and detailed reports have been categorized into the outputs/random_forest_master/ directory.*")
            
        print(f"\n[SUCCESS] Random Forest Master pipeline runs complete. Consolidated report saved to: {report_path}")

if __name__ == "__main__":
    main()
