import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from sklearn.neighbors import KNeighborsClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set path and device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training acceleration device: {DEVICE}")

# Add path for models import
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from high_capacity_research.models import (
    UnimodalExpert, EarlyFusionModel, GatedFusionModel,
    CrossAttentionFusionModel, HybridMoEAttentionModel
)

# Output directory for results
OUTPUT_DIR = os.path.join(backend_dir, "loso_evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helpers to extract sub-modality slices for deep models
# ---------------------------------------------------------
def get_modality_slices(df, dual=False):
    suffix = "" if not dual else "_abs"
    
    # 1. Face sub-features
    eye_cols = [f"face_ear_mean{suffix}"]
    mouth_cols = [f"face_mar_mean{suffix}"]
    gface_cols = [f"face_brow_mean{suffix}"] + [f"face_deep_embed_{i}{suffix}" for i in range(1, 513) if f"face_deep_embed_{i}{suffix}" in df.columns]
    
    # 2. Voice sub-features
    prosody_cols = [f"voice_rms_mean{suffix}", f"voice_zcr_mean{suffix}", f"voice_pitch_mean{suffix}", f"voice_pitch_std{suffix}"]
    spectral_cols = [f"voice_mfcc_{i}{suffix}" for i in range(1, 14) if f"voice_mfcc_{i}{suffix}" in df.columns]
    quality_cols = [f"quality_score{suffix}", f"face_confidence{suffix}", f"physio_continuity_flag{suffix}"]
    
    # 3. Physio sub-features
    cardio_cols = [f"ecg_hr{suffix}", f"ecg_mean{suffix}", f"ecg_std{suffix}", f"eda_tonic_mean{suffix}", f"eda_phasic_mean{suffix}"]
    motion_cols = [f"resp_rate_mean{suffix}", f"resp_std{suffix}"]
    
    # Padding functions to match expected PyTorch input shapes if columns are empty
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
# PyTorch DataLoader Setup for sequential learning
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
    # Repeat padding at start for sequence learning
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
# Trainer / Evaluator Core Module
# ---------------------------------------------------------
def train_and_eval_pytorch_model(model_name, make_model_fn, train_loader, val_loader, epochs=5, is_adversarial=False):
    model = make_model_fn().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_stress = nn.CrossEntropyLoss()
    criterion_subj = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Map batch keys to model inputs
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
                    stress_logits = model(*inputs[-3:]) # early fusion expects face_x, voice_x, physio_x
                elif model_name.startswith("Unimodal"):
                    stress_logits = model(inputs[0]) # unimodal encoder expects x
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
# Main Execution Module
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Master Stress Model Benchmarking Suite")
    parser.add_argument("--mode", type=str, default="group_kfold", choices=["random_split", "group_kfold", "full_loso"],
                        help="Validation mode to run: random_split, group_kfold (5-fold, fast LOSO), or full_loso (65-fold)")
    args = parser.parse_args()
    
    feature_file = os.path.join(OUTPUT_DIR, "stress_features_fusion_5s.csv")
    if not os.path.exists(feature_file):
        print(f"Error: {feature_file} not found. Please run feature extraction first.")
        sys.exit(1)
        
    print(f"Loading feature store: {feature_file}...")
    df = pd.read_csv(feature_file)
    
    # Filter out NaNs in labels
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    # Map subject labels to integers for classification
    subj_list = df["subject_id"].unique().tolist()
    subj_map = {s: i for i, s in enumerate(subj_list)}
    subj_indices = df["subject_id"].map(subj_map).values
    
    labels = df["label"].astype(int).values
    subjects = df["subject_id"].values
    
    # Slices for classical model (exclude identifiers and metadata)
    exclude_cols = ["subject_id", "task_id", "window_index", "label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_abs")]
    X_classical = StandardScaler().fit_transform(df[feature_cols].fillna(0).values)
    
    # Extract modality slices for sequential encoders
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
    
    # Setup cross validation folds
    if args.mode == "random_split":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X_classical, labels))
        print("Using Stratified 5-Fold Random Split validation.")
    elif args.mode == "group_kfold":
        cv = GroupKFold(n_splits=5)
        splits = list(cv.split(X_classical, labels, groups=subjects))
        print("Using Group 5-Fold (Subject-independent) validation.")
    else:
        # Full LOSO (65 folds)
        cv = GroupKFold(n_splits=len(subj_list))
        splits = list(cv.split(X_classical, labels, groups=subjects))
        print(f"Using full Leave-One-Subject-Out ({len(subj_list)} folds) validation.")
        
    # Dictionary to hold final evaluation results
    summary_results = []
    
    # ---------------------------------------------------------
    # Helper to calculate and store model stats
    # ---------------------------------------------------------
    def evaluate_predictions(name, category, targets, preds, probs):
        acc = accuracy_score(targets, preds)
        prec = precision_score(targets, preds, average="binary", zero_division=0)
        rec = recall_score(targets, preds, average="binary", zero_division=0)
        f1 = f1_score(targets, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(targets, probs)
        except ValueError:
            auc = 0.5
            
        summary_results.append({
            "Model Name": name,
            "Category": category,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        print(f"[{name}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Save plots for top performers
        if name in ["VBC-CASA-IS", "SSVB-CASA-AIS", "XGBoost", "Random Forest"]:
            # ROC curves
            fpr, tpr, _ = roc_curve(targets, probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name.lower().replace('-', '_')}_roc.png"))
            plt.close()
            
            # Confusion matrices
            cm = confusion_matrix(targets, preds)
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {name}')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Calm', 'Stress'])
            plt.yticks(tick_marks, ['Calm', 'Stress'])
            
            # Labeling cells
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name.lower().replace('-', '_')}_cm.png"))
            plt.close()

    # =========================================================
    # Part 1: Train & Evaluate Classical ML Models
    # =========================================================
    # Configure GPU-accelerated XGBoost if available
    xgb_clf = None
    if HAS_XGBOOST:
        # Try XGBoost 2.0+ GPU syntax
        try:
            xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", tree_method="hist", device="cuda")
            # Fit on a tiny dummy array to verify GPU driver support
            xgb_clf.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
            print("[INFO] XGBoost GPU acceleration initialized (device='cuda').")
        except Exception:
            # Try legacy XGBoost GPU syntax
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

    print("\n--- Training Classical ML Models ---")
    classical_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM": SVC(probability=True, class_weight="balanced", max_iter=2000, cache_size=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1),
        "XGBoost": xgb_clf,
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    for name, clf in classical_models.items():
        all_targets, all_preds, all_probs = [], [], []
        for train_idx, val_idx in splits:
            X_train, X_val = X_classical[train_idx], X_classical[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)
            probs = clf.predict_proba(X_val)[:, 1]
            
            all_targets.extend(y_val)
            all_preds.extend(preds)
            all_probs.extend(probs)
            
        evaluate_predictions(name, "Classical", np.array(all_targets), np.array(all_preds), np.array(all_probs))

    factor = 2

    # =========================================================
    # Part 2: Train & Evaluate Unimodal Sequence Experts (Deep)
    # =========================================================
    print("\n--- Training Unimodal sequence Experts ---")
    
    # 1. Unimodal Face Expert
    print("Evaluating Face Expert...")
    face_inputs = {"eye": seq_data["eye"], "mouth": seq_data["mouth"], "global_face": seq_data["global_face"]}
    targets_f, preds_f, probs_f = [], [], []
    for train_idx, val_idx in splits:
        train_d = {k: v[train_idx] for k, v in face_inputs.items()}
        val_d = {k: v[val_idx] for k, v in face_inputs.items()}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "UnimodalFace",
            lambda: UnimodalExpert(input_dim=5 * factor, hidden_dim=16, adversarial=False),
            train_loader, val_loader
        )
        targets_f.extend(t)
        preds_f.extend(p)
        probs_f.extend(pr)
    evaluate_predictions("Face Sequence Expert", "Unimodal Expert", np.array(targets_f), np.array(preds_f), np.array(probs_f))

    # 2. Unimodal Voice Expert
    print("Evaluating Voice Expert...")
    voice_inputs = {"prosody": seq_data["prosody"], "spectral": seq_data["spectral"], "quality": seq_data["quality"]}
    targets_v, preds_v, probs_v = [], [], []
    for train_idx, val_idx in splits:
        train_d = {k: v[train_idx] for k, v in voice_inputs.items()}
        val_d = {k: v[val_idx] for k, v in voice_inputs.items()}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "UnimodalVoice",
            lambda: UnimodalExpert(input_dim=3 * factor, hidden_dim=16, adversarial=False),
            train_loader, val_loader
        )
        targets_v.extend(t)
        preds_v.extend(p)
        probs_v.extend(pr)
    evaluate_predictions("Voice Sequence Expert", "Unimodal Expert", np.array(targets_v), np.array(preds_v), np.array(probs_v))

    # =========================================================
    # Part 3: Train & Evaluate Intermediate Fusion Models
    # =========================================================
    print("\n--- Training Multimodal Fusion & Gated Routers ---")
    
    # Early Concat Fusion
    print("Evaluating Early Fusion...")
    targets_ef, preds_ef, probs_ef = [], [], []
    for train_idx, val_idx in splits:
        train_d = {"eye": seq_data["global_face"][train_idx], "mouth": seq_data["spectral"][train_idx], "global_face": seq_data["cardio"][train_idx]}
        val_d = {"eye": seq_data["global_face"][val_idx], "mouth": seq_data["spectral"][val_idx], "global_face": seq_data["cardio"][val_idx]}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "EarlyFusion",
            lambda: EarlyFusionModel(face_dim=8 * factor, voice_dim=2 * factor, physio_dim=3 * factor, hidden_dim=16),
            train_loader, val_loader
        )
        targets_ef.extend(t)
        preds_ef.extend(p)
        probs_ef.extend(pr)
    evaluate_predictions("Early Concat Fusion", "Early Fusion", np.array(targets_ef), np.array(preds_ef), np.array(probs_ef))

    # Gated Fusion
    print("Evaluating Gated Fusion...")
    targets_gf, preds_gf, probs_gf = [], [], []
    for train_idx, val_idx in splits:
        train_d = {"eye": seq_data["global_face"][train_idx], "mouth": seq_data["spectral"][train_idx], "global_face": seq_data["cardio"][train_idx]}
        val_d = {"eye": seq_data["global_face"][val_idx], "mouth": seq_data["spectral"][val_idx], "global_face": seq_data["cardio"][val_idx]}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "GatedFusion",
            lambda: GatedFusionModel(face_dim=8 * factor, voice_dim=2 * factor, physio_dim=3 * factor, hidden_dim=16),
            train_loader, val_loader
        )
        targets_gf.extend(t)
        preds_gf.extend(p)
        probs_gf.extend(pr)
    evaluate_predictions("Gated Fusion", "Early Fusion", np.array(targets_gf), np.array(preds_gf), np.array(probs_gf))

    # =========================================================
    # Part 4: Train & Evaluate Production Models (VBC-CASA-IS)
    # =========================================================
    print("\n--- Training Production Models ---")
    
    # VBC-CASA-IS (No GRL adversarial identity suppression)
    print("Evaluating VBC-CASA-IS...")
    targets_vbc, preds_vbc, probs_vbc = [], [], []
    for train_idx, val_idx in splits:
        train_d = {k: v[train_idx] for k, v in seq_data.items()}
        val_d = {k: v[val_idx] for k, v in seq_data.items()}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "VBC-CASA-IS",
            lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=len(subj_list), adversarial=False, dual_representation=True),
            train_loader, val_loader
        )
        targets_vbc.extend(t)
        preds_vbc.extend(p)
        probs_vbc.extend(pr)
    evaluate_predictions("VBC-CASA-IS", "Production", np.array(targets_vbc), np.array(preds_vbc), np.array(probs_vbc))

    # SSVB-CASA-AIS (With GRL adversarial identity suppression)
    print("Evaluating SSVB-CASA-AIS...")
    targets_ssvbc, preds_ssvbc, probs_ssvbc = [], [], []
    for train_idx, val_idx in splits:
        train_d = {k: v[train_idx] for k, v in seq_data.items()}
        val_d = {k: v[val_idx] for k, v in seq_data.items()}
        
        train_ds = SeqMultimodalDataset(train_d, labels[train_idx], subjects=subj_indices[train_idx])
        val_ds = SeqMultimodalDataset(val_d, labels[val_idx], subjects=subj_indices[val_idx])
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        
        t, p, pr = train_and_eval_pytorch_model(
            "SSVB-CASA-AIS",
            lambda: HybridMoEAttentionModel(hidden_dim=16, num_subjects=len(subj_list), adversarial=True, dual_representation=True),
            train_loader, val_loader,
            is_adversarial=True
        )
        targets_ssvbc.extend(t)
        preds_ssvbc.extend(p)
        probs_ssvbc.extend(pr)
    evaluate_predictions("SSVB-CASA-AIS", "Production", np.array(targets_ssvbc), np.array(preds_ssvbc), np.array(probs_ssvbc))

    # Save final leaderboard table
    df_results = pd.DataFrame(summary_results)
    df_results = df_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Save CSV
    leaderboard_path = os.path.join(OUTPUT_DIR, "loso_leaderboard.csv")
    df_results.to_csv(leaderboard_path, index=False)
    print(f"\nLeaderboard CSV saved to: {leaderboard_path}")
    
    # Save Markdown report
    report_path = os.path.join(OUTPUT_DIR, "loso_report.md")
    with open(report_path, "w") as f:
        f.write("# Leave-One-Subject-Out (LOSO) Stress Detection Leaderboard\n\n")
        f.write(f"**Validation Mode:** {args.mode.upper()}\n")
        f.write(f"**Feature File Ingested:** {feature_file}\n")
        f.write(f"**Total Records:** {len(df)}\n")
        f.write(f"**Unique Subjects:** {len(subj_list)}\n\n")
        
        # Build markdown table manually to avoid tabulate dependency
        headers = list(df_results.columns)
        md_table = "| " + " | ".join(headers) + " |\n"
        md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df_results.iterrows():
            vals = []
            for val in row:
                if isinstance(val, float):
                    vals.append(f"{val:.4f}")
                else:
                    vals.append(str(val))
            md_table += "| " + " | ".join(vals) + " |\n"
            
        f.write(md_table)
        f.write("\n\n*Plots for top performing models have been generated inside the results directory.*")
        
    print(f"Leaderboard Markdown report saved to: {report_path}")
    print("All training metrics completed successfully!")

if __name__ == "__main__":
    main()
