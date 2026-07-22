import os
import sys
import time
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# Seed setting for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Acceleration device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Hardware Acceleration: {DEVICE}")

# Add backend root to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
sys.path.append(os.path.join(backend_dir, "research", "Phase_2_High_Capacity"))

from models import (
    UnimodalExpert, EarlyFusionModel, GatedFusionModel,
    CrossAttentionFusionModel, HybridMoEAttentionModel
)

# Detect XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ---------------------------------------------------------
# Step 1: Folder Structure Setup
# ---------------------------------------------------------
BASE_DIR = os.path.join(backend_dir, "research", "Phase_1_Baseline_LOSO")
os.makedirs(os.path.join(BASE_DIR, "configs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data_links"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models", "classical"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models", "unimodal_deep"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models", "fusion"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models", "production"), exist_ok=True)

for scale in ["2sec", "5sec", "10sec"]:
    os.makedirs(os.path.join(BASE_DIR, "outputs", scale), exist_ok=True)

# ---------------------------------------------------------
# Step 2: Feature Slicing Helper
# ---------------------------------------------------------
def get_modality_slices(df, dual=False):
    suffix = "" if not dual else "_abs"
    
    eye_cols = [f"face_ear_mean{suffix}"]
    mouth_cols = [f"face_mar_mean{suffix}"]
    gface_cols = [f"face_brow_mean{suffix}"] + [f"face_deep_embed_{i}{suffix}" for i in range(1, 513) if f"face_deep_embed_{i}{suffix}" in df.columns]
    
    prosody_cols = [f"voice_rms_mean{suffix}", f"voice_zcr_mean{suffix}", f"voice_pitch_mean{suffix}", f"voice_pitch_std{suffix}"]
    spectral_cols = [f"voice_mfcc_{i}{suffix}" for i in range(1, 14) if f"voice_mfcc_{i}{suffix}" in df.columns]
    quality_cols = [f"quality_score{suffix}", f"face_confidence{suffix}", f"physio_continuity_flag{suffix}"]
    
    cardio_cols = [f"ecg_hr{suffix}", f"ecg_mean{suffix}", f"ecg_std{suffix}", f"eda_tonic_mean{suffix}", f"eda_phasic_mean{suffix}"]
    motion_cols = [f"resp_rate_mean{suffix}", f"resp_std{suffix}"]
    
    def safe_slice(cols, target_dim):
        existing = [c for c in cols if c in df.columns]
        if len(existing) == 0:
            return np.zeros((len(df), target_dim), dtype=np.float32)
        arr = df[existing].fillna(0).values
        if arr.shape[1] < target_dim:
            pad = np.zeros((len(df), target_dim - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad])
        return arr[:, :target_dim]

    factor = 2 if dual else 1
    eye_arr = safe_slice(eye_cols, 5 * factor)
    mouth_arr = safe_slice(mouth_cols, 3 * factor)
    gface_arr = safe_slice(gface_cols, 8 * factor)
    
    prosody_arr = safe_slice(prosody_cols, 3 * factor)
    spectral_arr = safe_slice(spectral_cols, 2 * factor)
    quality_arr = safe_slice(quality_cols, 5 * factor)
    
    cardio_arr = safe_slice(cardio_cols, 3 * factor)
    motion_arr = safe_slice(motion_cols, 1 * factor)
    
    return eye_arr, mouth_arr, gface_arr, prosody_arr, spectral_arr, quality_arr, cardio_arr, motion_arr

# ---------------------------------------------------------
# Step 3: Sequence Helpers
# ---------------------------------------------------------
class SeqMultimodalDataset(Dataset):
    def __init__(self, data_dict, labels, subjects=None):
        self.labels = torch.LongTensor(labels)
        self.subjects = torch.LongTensor(subjects) if subjects is not None else None
        self.data = {}
        for k, v in data_dict.items():
            self.data[k] = torch.FloatTensor(v)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.data.items()}
        item["label"] = self.labels[idx]
        if self.subjects is not None:
            item["subject"] = self.subjects[idx]
        return item

def make_sequences(arr, seq_len=5):
    N, D = arr.shape
    seqs = []
    for i in range(N):
        if i < seq_len - 1:
            pad = np.repeat(arr[i:i+1], seq_len - 1 - i, axis=0)
            seq = np.vstack([pad, arr[0:i+1]])
        else:
            seq = arr[i - seq_len + 1 : i + 1]
        seqs.append(seq)
    return np.array(seqs)

# ---------------------------------------------------------
# Step 4: PyTorch Model Training & Evaluation Helper
# ---------------------------------------------------------
def train_and_eval_pytorch_model(model_name, make_model_fn, train_loader, val_loader, epochs=3, is_adversarial=False):
    model = make_model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_stress = nn.CrossEntropyLoss()
    criterion_subj = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = []
            for k in ["eye", "mouth", "global_face", "prosody", "spectral", "quality", "cardio", "motion"]:
                if k in batch:
                    inputs.append(batch[k].to(DEVICE))
                    
            labels = batch["label"].to(DEVICE)
            
            if is_adversarial:
                subjects = batch["subject"].to(DEVICE)
                stress_logits, subj_logits = model(*inputs)
                loss = criterion_stress(stress_logits, labels) + 0.1 * criterion_subj(subj_logits, subjects)
            else:
                if len(inputs) == 3 and model_name in ["EarlyFusion", "GatedFusion", "CrossAttentionFusion"]:
                    stress_logits = model(*inputs[-3:])
                elif model_name.startswith("Unimodal"):
                    stress_logits = model(inputs[0])
                else:
                    stress_logits = model(*inputs)
                loss = criterion_stress(stress_logits, labels)
                
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = []
            for k in ["eye", "mouth", "global_face", "prosody", "spectral", "quality", "cardio", "motion"]:
                if k in batch:
                    inputs.append(batch[k].to(DEVICE))
            labels = batch["label"].to(DEVICE)
            
            if is_adversarial:
                stress_logits, _ = model(*inputs)
            else:
                if len(inputs) == 3 and model_name in ["EarlyFusion", "GatedFusion", "CrossAttentionFusion"]:
                    stress_logits = model(*inputs[-3:])
                elif model_name.startswith("Unimodal"):
                    stress_logits = model(inputs[0])
                else:
                    stress_logits = model(*inputs)
                    
            probs = torch.softmax(stress_logits, dim=1)[:, 1].cpu().numpy()
            preds = stress_logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

# ---------------------------------------------------------
# Step 5: Master Pipeline Execution
# ---------------------------------------------------------
def main():
    # Setup GPU-accelerated XGBoost if available
    xgb_clf = None
    if HAS_XGBOOST:
        try:
            xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="hist", device="cuda")
            xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
            print("[INFO] XGBoost GPU acceleration initialized (device='cuda').")
        except Exception:
            try:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="gpu_hist")
                xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
                print("[INFO] XGBoost GPU acceleration initialized (tree_method='gpu_hist').")
            except Exception:
                xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                print("[INFO] XGBoost GPU initialization failed. Running on CPU.")
    else:
        xgb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5)
        print("[INFO] XGBoost not installed. Using sklearn Gradient Boosting on CPU.")

    dataset_scales = [
        ("2sec", "stress_features_fusion_2s.csv"),
        ("5sec", "stress_features_fusion_5s.csv"),
        ("10sec", "stress_features_fusion_10s.csv")
    ]
    
    global_leaderboard = []

    for scale, filename in dataset_scales:
        print(f"\n==========================================================")
        print(f"  PROCESSING DATASET SCALE: {scale} ({filename})")
        print(f"==========================================================\n")
        
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            # Try workspace root
            file_path = os.path.join(backend_dir, filename)
            if not os.path.exists(file_path):
                print(f"[WARNING] Feature file {filename} not found. Skipping scale {scale}.")
                continue
                
        # Link feature files in data_links for documentation
        link_path = os.path.join(BASE_DIR, "data_links", f"{scale}_fusion_features.csv")
        if not os.path.exists(link_path):
            try:
                if sys.platform == "win32":
                    with open(link_path, "w") as f:
                        f.write(f"Linked Source: {file_path}\n")
                else:
                    os.symlink(file_path, link_path)
            except Exception:
                pass

        print(f"Loading feature store: {file_path}...")
        df = pd.read_csv(file_path)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        
        subj_list = df["subject_id"].unique().tolist()
        subj_map = {s: i for i, s in enumerate(subj_list)}
        subj_indices = df["subject_id"].map(subj_map).values
        
        labels = df["label"].astype(int).values
        subjects = df["subject_id"].values
        
        # Setup Classical ML Features
        exclude_cols = ["subject_id", "task_id", "window_index", "label"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_abs")]
        X_classical = StandardScaler().fit_transform(df[feature_cols].fillna(0).values)
        
        # Extract modality slices for sequence models
        eye, mouth, gface, prosody, spectral, quality, cardio, motion = get_modality_slices(df, dual=True)
        
        # Form sequence inputs (seq_len = 5)
        seq_len = 5
        seq_data = {
            "eye": make_sequences(eye, seq_len),
            "mouth": make_sequences(mouth, seq_len),
            "global_face": make_sequences(gface, seq_len),
            "prosody": make_sequences(prosody, seq_len),
            "spectral": make_sequences(spectral, seq_len),
            "quality": make_sequences(quality, seq_len),
            "cardio": make_sequences(cardio, seq_len),
            "motion": make_sequences(motion, seq_len)
        }
        
        # Group K-Fold validation to guarantee subject-independence
        cv = GroupKFold(n_splits=5)
        splits = list(cv.split(X_classical, labels, groups=subjects))
        
        # Config saving
        config_path = os.path.join(BASE_DIR, "configs", f"config_{scale}.json")
        with open(config_path, "w") as f:
            json.dump({
                "scale": scale,
                "num_records": len(df),
                "num_subjects": len(subj_list),
                "features_count": len(feature_cols),
                "feature_names": feature_cols,
                "split_strategy": "5-Fold GroupKFold (subject independent)"
            }, f, indent=4)
            
        models_to_run = {}
        
        # 1. Classical Models
        classical_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "SVM": SVC(probability=True, class_weight="balanced", max_iter=2000, cache_size=2000),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1),
            "XGBoost": xgb_clf,
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        for n, m in classical_models.items():
            models_to_run[n] = ("classical", "CPU", lambda m=m: m)
            
        # 2. Deep & Fusion Models
        factor = 2
        models_to_run["FaceSequenceExpert"] = ("unimodal_deep", "GPU", lambda: UnimodalExpert(input_dim=5 * factor, hidden_dim=16, adversarial=False))
        models_to_run["VoiceSequenceExpert"] = ("unimodal_deep", "GPU", lambda: UnimodalExpert(input_dim=3 * factor, hidden_dim=16, adversarial=False))
        models_to_run["EarlyConcatFusion"] = ("fusion", "GPU", lambda: EarlyFusionModel(face_dim=8 * factor, voice_dim=2 * factor, physio_dim=3 * factor, hidden_dim=16))
        models_to_run["GatedFusion"] = ("fusion", "GPU", lambda: GatedFusionModel(face_dim=8 * factor, voice_dim=2 * factor, physio_dim=3 * factor, hidden_dim=16))
        models_to_run["VBC_CASA_IS"] = ("production", "GPU", lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=len(subj_list), adversarial=False, dual_representation=True))
        models_to_run["SSVB_CASA_AIS"] = ("production", "GPU", lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=len(subj_list), adversarial=True, dual_representation=True))
        
        # Process each model
        for model_name, (category, hardware, make_fn) in models_to_run.items():
            print(f"--> Training {model_name} ({category} on {hardware})...")
            start_time = time.time()
            
            all_targets, all_preds, all_probs = [], [], []
            fold_metrics = []
            
            # Cross-validation loops
            fold_idx = 1
            for train_idx, val_idx in splits:
                if hardware == "CPU":
                    clf = make_fn()
                    clf.fit(X_classical[train_idx], labels[train_idx])
                    preds = clf.predict(X_classical[val_idx])
                    probs = clf.predict_proba(X_classical[val_idx])[:, 1]
                    t_fold = labels[val_idx]
                else:
                    # Extract only the relevant modal keys for this specific architecture to avoid contamination
                    if model_name == "FaceSequenceExpert":
                        keys = ["eye"]
                    elif model_name == "VoiceSequenceExpert":
                        keys = ["prosody"]
                    elif model_name in ["EarlyConcatFusion", "GatedFusion"]:
                        keys = ["global_face", "spectral", "cardio"]
                    else:
                        keys = list(seq_data.keys())
                        
                    train_d = {k: seq_data[k][train_idx] for k in keys}
                    val_d = {k: seq_data[k][val_idx] for k in keys}
                    
                    # Handle subjects mapping for adversarial model
                    subj_in = subj_indices[train_idx] if model_name == "SSVB_CASA_AIS" else None
                    
                    train_ds = SeqMultimodalDataset(train_d, labels[train_idx], subjects=subj_in)
                    val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
                    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
                    
                    t_fold, preds, probs = train_and_eval_pytorch_model(
                        model_name,
                        make_fn,
                        train_loader,
                        val_loader,
                        epochs=3,
                        is_adversarial=(model_name == "SSVB_CASA_AIS")
                    )
                
                # Compute fold metrics
                fold_acc = accuracy_score(t_fold, preds)
                fold_f1 = f1_score(t_fold, preds, average="binary", zero_division=0)
                try:
                    fold_auc = roc_auc_score(t_fold, probs)
                except ValueError:
                    fold_auc = 0.5
                    
                fold_metrics.append({
                    "Fold": fold_idx,
                    "Accuracy": fold_acc,
                    "F1-Score": fold_f1,
                    "ROC-AUC": fold_auc
                })
                
                all_targets.extend(t_fold)
                all_preds.extend(preds)
                all_probs.extend(probs)
                fold_idx += 1
                
            elapsed_time = time.time() - start_time
            
            # Overall evaluation metrics
            all_targets = np.array(all_targets)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            
            acc = accuracy_score(all_targets, all_preds)
            prec = precision_score(all_targets, all_preds, average="binary", zero_division=0)
            rec = recall_score(all_targets, all_preds, average="binary", zero_division=0)
            f1 = f1_score(all_targets, all_preds, average="binary", zero_division=0)
            bal_acc = balanced_accuracy_score(all_targets, all_preds)
            try:
                auc = roc_auc_score(all_targets, all_probs)
            except ValueError:
                auc = 0.5
                
            # Create model output directory
            model_out_dir = os.path.join(BASE_DIR, "outputs", scale, model_name)
            os.makedirs(model_out_dir, exist_ok=True)
            
            # Save predictions
            pd.DataFrame({
                "Actual": all_targets,
                "Predicted": all_preds,
                "Probability": all_probs
            }).to_csv(os.path.join(model_out_dir, "predictions.csv"), index=False)
            
            # Save fold metrics
            df_folds = pd.DataFrame(fold_metrics)
            df_folds.to_csv(os.path.join(model_out_dir, "fold_results.csv"), index=False)
            
            # Save overall metrics
            pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced-Accuracy", "ROC-AUC", "Runtime-Seconds"],
                "Value": [acc, prec, rec, f1, bal_acc, auc, elapsed_time]
            }).to_csv(os.path.join(model_out_dir, "metrics.csv"), index=False)
            
            # Generate Confusion Matrix
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "confusion_matrix.png"))
            plt.close()
            
            # Generate ROC curve
            fpr, tpr, _ = roc_curve(all_targets, all_probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "roc_curve.png"))
            plt.close()
            
            # Generate PR curve
            pr_y, pr_x, _ = precision_recall_curve(all_targets, all_probs)
            plt.figure()
            plt.plot(pr_x, pr_y, label='Precision-Recall curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curve - {model_name}')
            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(os.path.join(model_out_dir, "pr_curve.png"))
            plt.close()
            
            # Save summary report
            with open(os.path.join(model_out_dir, "summary.md"), "w") as f:
                f.write(f"# Performance Summary: {model_name} ({scale})\n\n")
                f.write(f"* **Category:** {category}\n")
                f.write(f"* **Hardware Execution:** {hardware}\n")
                f.write(f"* **Execution Runtime:** {elapsed_time:.2f} seconds\n\n")
                f.write(f"### Overall Evaluation Metrics\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"| --- | --- |\n")
                f.write(f"| Accuracy | {acc:.4f} |\n")
                f.write(f"| Precision | {prec:.4f} |\n")
                f.write(f"| Recall | {rec:.4f} |\n")
                f.write(f"| F1-Score | {f1:.4f} |\n")
                f.write(f"| Balanced Accuracy | {bal_acc:.4f} |\n")
                f.write(f"| ROC-AUC | {auc:.4f} |\n")
                
            # Log in global leaderboard
            global_leaderboard.append({
                "Scale": scale,
                "Model Name": model_name,
                "Category": category,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": auc,
                "Runtime-Seconds": elapsed_time
            })
            
    # Save global reports
    df_global = pd.DataFrame(global_leaderboard)
    df_global = df_global.sort_values(by=["Scale", "Accuracy"], ascending=[True, False]).reset_index(drop=True)
    df_global.to_csv(os.path.join(BASE_DIR, "metrics", "pipeline_leaderboard.csv"), index=False)
    
    # Save final report markdown
    report_path = os.path.join(BASE_DIR, "reports", "final_summary_report.md")
    with open(report_path, "w") as f:
        f.write("# Master Model Benchmarking final Summary Report\n\n")
        f.write("This report compiles performance comparisons across all window scales (2s, 5s, 10s) and model architectures.\n\n")
        
        # Build markdown table manually to avoid tabulate dependency
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
        f.write("\n\n*All plots and detailed reports have been categorized into the outputs/ directory.*")
        
    print(f"\n[SUCCESS] Pipeline runs complete. Final report saved to: {report_path}")

if __name__ == "__main__":
    main()
