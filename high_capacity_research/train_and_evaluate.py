import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, confusion_matrix
)

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from high_capacity_research.data_loader import load_and_align_data, MultimodalExpertDataset
from high_capacity_research.models import (
    UnimodalExpert, EarlyFusionModel, GatedFusionModel, 
    CrossAttentionFusionModel, HybridMoEAttentionModel
)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Configuration
EPOCHS = 8
BATCH_SIZE = 256
SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS_DIR = os.path.join(backend_dir, "evaluation_reports", "high_capacity_research")
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ---------------------------------------------------------
# Evaluation Metric Computations
# ---------------------------------------------------------
def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    mse = mean_squared_error(y_true, y_prob)
    mae = mean_absolute_error(y_true, y_prob)
    r2 = r2_score(y_true, y_prob)
    rmse = np.sqrt(mse)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "MSE": mse,
        "MAE": mae,
        "R2-Score": r2,
        "RMSE": rmse
    }

# ---------------------------------------------------------
# Plotting and Report Generation
# ---------------------------------------------------------
def generate_visual_reports(model_name, y_true, y_prob, folds_metrics):
    model_dir = os.path.join(REPORTS_DIR, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. ROC-AUC Curve
    plt.figure(figsize=(6, 5), dpi=150)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='#2b5c8f', lw=2, label=f"5-Fold CV (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color='#999999', linestyle='--', lw=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=11, fontweight='bold')
    plt.title(f"ROC-AUC Curve - {model_name}", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roc_auc_curves.png"), bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    plt.figure(figsize=(5, 4), dpi=150)
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    labels = np.array([
        [f"TN\n{cm[0,0]}\n({cm_norm[0,0]*100:.1f}%)", f"FP\n{cm[0,1]}\n({cm_norm[0,1]*100:.1f}%)"],
        [f"FN\n{cm[1,0]}\n({cm_norm[1,0]*100:.1f}%)", f"TP\n{cm[1,1]}\n({cm_norm[1,1]*100:.1f}%)"]
    ])
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, annot_kws={"size": 11, "weight": "bold"})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {model_name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrices.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Dashboard summary
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.axis('off')
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MSE", "MAE", "R2-Score", "RMSE"]
    header = ["Fold"] + metrics_list
    
    table_data = []
    for idx, fold in enumerate(folds_metrics):
        row = [f"Fold {idx+1}"]
        for m in metrics_list:
            row.append(f"{fold[m]:.4f}")
        table_data.append(row)
        
    mean_row = ["Mean"]
    for m in metrics_list:
        mean_row.append(f"{np.mean([f[m] for f in folds_metrics]):.4f}")
    table_data.append(mean_row)
    
    table = ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    
    for col_idx in range(len(header)):
        cell = table[0, col_idx]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1e3d59')
        
    for row_idx in range(1, len(table_data) + 1):
        face_color = '#f5f5f5' if row_idx % 2 == 0 else 'white'
        if row_idx == len(table_data):
            face_color = '#e0f2f1'
        for col_idx in range(len(header)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor(face_color)
            if col_idx == 0 or row_idx == len(table_data):
                cell.set_text_props(weight='bold')
                
    ax.set_title(f"Model Performance Metrics - {model_name}", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "metrics_comparison.png"), bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(folds_metrics, f, indent=4)
        
    print(f"  -> Generated visualization reports under {model_dir}")

# ---------------------------------------------------------
# Training and Evaluation Harness
# ---------------------------------------------------------
def run_evaluation_loop(model_name, model_type, model_builder, df_merged, groups, gkf):
    print(f"\n==========================================")
    print(f"Evaluating: {model_name}")
    print(f"==========================================")
    
    results = []
    preds_y_true = []
    preds_y_prob = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df_merged, df_merged['label'], groups)):
        print(f"  -> Fold {fold+1}/5 training...")
        
        # Load datasets
        train_ds = MultimodalExpertDataset(df_merged.iloc[train_idx])
        test_ds = MultimodalExpertDataset(df_merged.iloc[test_idx],
                                         face_scaler=train_ds.face_scaler,
                                         voice_scaler=train_ds.voice_scaler,
                                         physio_scaler=train_ds.physio_scaler)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = model_builder().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion_stress = nn.CrossEntropyLoss()
        
        # Check if adversarial
        adversarial = (model_type == "adversarial")
        criterion_subject = nn.CrossEntropyLoss() if adversarial else None
        
        # Train
        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                eye, mouth, gface, prosody, spectral, quality, cardio, motion, label, subj_id = [x.to(DEVICE) for x in batch]
                
                optimizer.zero_grad()
                
                # Model inputs mapping
                if model_type == "face":
                    inputs = torch.cat([eye, mouth, gface], dim=2)
                    logits = model(inputs)
                elif model_type == "voice":
                    inputs = torch.cat([prosody, spectral, quality], dim=2)
                    logits = model(inputs)
                elif model_type == "physio":
                    inputs = torch.cat([cardio, motion], dim=2)
                    logits = model(inputs)
                elif model_type == "fusion":
                    fx = torch.cat([eye, mouth, gface], dim=2)
                    vx = torch.cat([prosody, spectral, quality], dim=2)
                    px = torch.cat([cardio, motion], dim=2)
                    logits = model(fx, vx, px)
                elif model_type == "hybrid":
                    logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                elif model_type == "adversarial":
                    logits, subj_logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                    
                # Loss computation
                if adversarial:
                    loss_stress = criterion_stress(logits, label)
                    loss_subj = criterion_subject(subj_logits, subj_id)
                    loss = loss_stress - 0.02 * loss_subj
                else:
                    loss = criterion_stress(logits, label)
                    
                loss.backward()
                optimizer.step()
                
        # Eval
        model.eval()
        fold_probs = []
        fold_trues = []
        
        with torch.no_grad():
            for batch in test_loader:
                eye, mouth, gface, prosody, spectral, quality, cardio, motion, label, _ = [x.to(DEVICE) for x in batch]
                
                if model_type == "face":
                    inputs = torch.cat([eye, mouth, gface], dim=2)
                    logits = model(inputs)
                elif model_type == "voice":
                    inputs = torch.cat([prosody, spectral, quality], dim=2)
                    logits = model(inputs)
                elif model_type == "physio":
                    inputs = torch.cat([cardio, motion], dim=2)
                    logits = model(inputs)
                elif model_type == "fusion":
                    fx = torch.cat([eye, mouth, gface], dim=2)
                    vx = torch.cat([prosody, spectral, quality], dim=2)
                    px = torch.cat([cardio, motion], dim=2)
                    logits = model(fx, vx, px)
                elif model_type == "hybrid" or model_type == "adversarial":
                    if adversarial:
                        logits, _ = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                    else:
                        logits = model(eye, mouth, gface, prosody, spectral, quality, cardio, motion)
                        
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                fold_probs.append(probs)
                fold_trues.append(label.cpu().numpy())
                
        fold_probs = np.hstack(fold_probs)
        fold_trues = np.hstack(fold_trues)
        
        results.append(calculate_metrics(fold_trues, fold_probs))
        preds_y_true.append(fold_trues)
        preds_y_prob.append(fold_probs)
        
    # Compile outputs
    preds_y_true = np.hstack(preds_y_true)
    preds_y_prob = np.hstack(preds_y_prob)
    
    mean_acc = np.mean([f["Accuracy"] for f in results])
    print(f"  -> Evaluation complete. Mean Accuracy: {mean_acc:.4f}")
    
    # Save visual reports
    generate_visual_reports(model_name, preds_y_true, preds_y_prob, results)
    return results

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------
def main():
    # Load and clean synchronized dataset
    df_merged = load_and_align_data()
    
    groups = df_merged['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    
    # Stage 1: Unimodal Experts
    run_evaluation_loop("Unimodal Face Expert", "face", 
                        lambda: UnimodalExpert(input_dim=16), df_merged, groups, gkf)
                        
    run_evaluation_loop("Unimodal Voice Expert", "voice", 
                        lambda: UnimodalExpert(input_dim=10), df_merged, groups, gkf)
                        
    run_evaluation_loop("Unimodal Physio Expert", "physio", 
                        lambda: UnimodalExpert(input_dim=4), df_merged, groups, gkf)
                        
    # Stage 2 & 3: Early & Gated Fusion Baselines
    run_evaluation_loop("Early Fusion Model", "fusion", 
                        lambda: EarlyFusionModel(face_dim=16, voice_dim=10, physio_dim=4), df_merged, groups, gkf)
                        
    run_evaluation_loop("Gated Fusion Model", "fusion", 
                        lambda: GatedFusionModel(face_dim=16, voice_dim=10, physio_dim=4), df_merged, groups, gkf)
                        
    # Stage 4: Cross-Attention Fusion Model
    run_evaluation_loop("Cross Attention Fusion Model", "fusion", 
                        lambda: CrossAttentionFusionModel(face_dim=16, voice_dim=10, physio_dim=4), df_merged, groups, gkf)
                        
    # Stage 5 & 6: Hybrid MoE + Cross-Attention Model (Standard & Adversarial)
    run_evaluation_loop("Hybrid MoE Attention Model", "hybrid", 
                        lambda: HybridMoEAttentionModel(adversarial=False), df_merged, groups, gkf)
                        
    run_evaluation_loop("Adversarial Hybrid MoE Attention Model", "adversarial", 
                        lambda: HybridMoEAttentionModel(adversarial=True), df_merged, groups, gkf)
                        
    print("\n==========================================")
    print("All research architectures evaluated successfully!")
    print(f"Results are cataloged inside: {REPORTS_DIR}")
    print("==========================================")

if __name__ == "__main__":
    main()
