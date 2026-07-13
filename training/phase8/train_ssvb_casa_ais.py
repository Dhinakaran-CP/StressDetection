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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, confusion_matrix
)

# Ensure backend root is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from high_capacity_research.data_loader import load_and_align_data, MultimodalExpertDataset
from high_capacity_research.models import HybridMoEAttentionModel

warnings.filterwarnings('ignore')

# Configuration
PRETRAIN_EPOCHS = 4
FINE_TUNE_EPOCHS = 8
BATCH_SIZE = 256
SEQ_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS_DIR = os.path.join(backend_dir, "evaluation_reports", "production_model", "ssvb_casa_ais")
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")

# ---------------------------------------------------------
# Self-Supervised Contrastive Learning Loss (Stage 1)
# ---------------------------------------------------------
def contrastive_loss(embeddings, subject_ids, temperature=0.1):
    """
    Computes InfoNCE contrastive loss where positive pairs are windows from
    the same subject/session and negative pairs are from different subjects.
    """
    norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Cosine similarity matrix shape: [batch, batch]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create mask for positive pairs (same subject, but not self)
    subject_ids = subject_ids.view(-1, 1)
    mask = torch.eq(subject_ids, subject_ids.T).float()
    
    # Exclude self-similarity
    diag_mask = torch.eye(mask.shape[0], device=embeddings.device)
    mask = mask - diag_mask
    
    exp_sim = torch.exp(sim_matrix)
    sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True) - torch.diag(exp_sim).view(-1, 1)
    
    log_prob = sim_matrix - torch.log(sum_exp_sim + 1e-8)
    
    pos_count = torch.sum(mask, dim=1)
    pos_count_safe = torch.where(pos_count > 0, pos_count, torch.ones_like(pos_count))
    
    loss = -torch.sum(log_prob * mask, dim=1) / pos_count_safe
    loss = torch.where(pos_count > 0, loss, torch.zeros_like(loss))
    
    return torch.mean(loss)

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
    model_dir = REPORTS_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. ROC-AUC Curve
    plt.figure(figsize=(6, 5), dpi=150)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='#1b7a60', lw=2, label=f"SSVB-CASA-AIS (AUC = {auc_val:.4f})")
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
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=11, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Calm (0)", "Stress (1)"])
    plt.yticks(tick_marks, ["Calm (0)", "Stress (1)"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Actual Label", fontsize=10, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrices.png"), bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# Main SSVB-CASA-AIS Training Loop
# ---------------------------------------------------------
def train_and_validate_loso(df_merged, groups, gkf):
    print("\n==========================================")
    print("Starting SSVB-CASA-AIS 5-Fold LOSO Pipeline")
    print("==========================================\n")
    
    num_subjects = len(np.unique(df_merged['subject_id']))
    folds_metrics = []
    preds_y_true = []
    preds_y_prob = []
    preds_y_conf = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_merged, groups=groups)):
        print(f"--- FOLD {fold+1}/5 ---")
        train_df = df_merged.iloc[train_idx].reset_index(drop=True)
        val_df = df_merged.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = MultimodalExpertDataset(train_df, seq_len=SEQ_LEN)
        val_dataset = MultimodalExpertDataset(val_df, 
                                              face_scaler=train_dataset.face_scaler, 
                                              voice_scaler=train_dataset.voice_scaler, 
                                              physio_scaler=train_dataset.physio_scaler,
                                              seq_len=SEQ_LEN)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = HybridMoEAttentionModel(adversarial=True, num_subjects=num_subjects).to(DEVICE)
        
        # ---------------------------------------------------------
        # STAGE 1: Self-Supervised Pretraining (Contrastive Learning)
        # ---------------------------------------------------------
        print("  Stage 1: Contrastive SSL Pretraining...")
        optimizer_pretrain = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            total_ssl_loss = 0.0
            for batch in train_loader:
                eye, mouth, gface, prosody, spectral, quality, cardio, motion, _, subj_id = [x.to(DEVICE) for x in batch]
                
                optimizer_pretrain.zero_grad()
                
                # Extract unimodal representations
                e_eye = model.exp_eye(eye)
                e_mouth = model.exp_mouth(mouth)
                e_gface = model.exp_global_face(gface)
                face_latent = torch.cat([e_eye, e_mouth, e_gface], dim=1)
                
                e_prosody = model.exp_prosody(prosody)
                e_spectral = model.exp_spectral(spectral)
                e_quality = model.exp_quality(quality)
                voice_latent = torch.cat([e_prosody, e_spectral, e_quality], dim=1)
                
                e_cardio = model.exp_cardio(cardio)
                e_motion = model.exp_motion(motion)
                physio_latent = torch.cat([e_cardio, e_motion], dim=1)
                
                # Compute multi-modality contrastive losses
                loss_f = contrastive_loss(face_latent, subj_id)
                loss_v = contrastive_loss(voice_latent, subj_id)
                loss_p = contrastive_loss(physio_latent, subj_id)
                
                loss_pretrain = loss_f + loss_v + loss_p
                loss_pretrain.backward()
                optimizer_pretrain.step()
                total_ssl_loss += loss_pretrain.item()
                
            print(f"    Epoch {epoch+1}/{PRETRAIN_EPOCHS} - SSL Loss: {total_ssl_loss/len(train_loader):.4f}")
            
        # ---------------------------------------------------------
        # STAGE 2-4: Supervised Fine-Tuning + Adversarial Identity + Gated Router Tuning
        # ---------------------------------------------------------
        print("  Stage 2-4: Supervised Fine-Tuning with Adversarial GRL & Dropout Routing...")
        optimizer_ft = optim.Adam(model.parameters(), lr=5e-4)
        criterion_subj = nn.CrossEntropyLoss()
        
        lambda_conf = 0.15
        lambda_subj = 0.10
        
        for epoch in range(FINE_TUNE_EPOCHS):
            model.train()
            total_supervised_loss = 0.0
            
            for batch in train_loader:
                eye, mouth, gface, prosody, spectral, quality, cardio, motion, label, subj_id = [x.to(DEVICE) for x in batch]
                
                # Stage 4: Sensor/Modality Dropout Gating Simulation
                # Randomly drop a modality (setting features to zero) with 15% probability
                dropout_choice = np.random.rand()
                if dropout_choice < 0.05:  # Drop Face
                    eye, mouth, gface = eye * 0, mouth * 0, gface * 0
                elif dropout_choice < 0.10: # Drop Voice
                    prosody, spectral, quality = prosody * 0, spectral * 0, quality * 0
                elif dropout_choice < 0.15: # Drop Physio
                    cardio, motion = cardio * 0, motion * 0
                
                optimizer_ft.zero_grad()
                
                # Forward return signature returns logits, subj_logits, and confidence score
                stress_logits, subj_logits, confidence = model(
                    eye, mouth, gface, prosody, spectral, quality, cardio, motion, 
                    return_confidence=True
                )
                
                # Stage 8: Confidence-Aware loss adaptation (DeVries et al.)
                probs = torch.softmax(stress_logits, dim=1)
                y_onehot = torch.nn.functional.one_hot(label, num_classes=2).float()
                
                probs_adj = confidence * probs + (1 - confidence) * y_onehot
                loss_stress_adj = -torch.sum(y_onehot * torch.log(probs_adj + 1e-8), dim=1).mean()
                loss_conf = -torch.log(confidence + 1e-8).mean()
                
                loss_supervised = loss_stress_adj + lambda_conf * loss_conf
                loss_subj = criterion_subj(subj_logits, subj_id)
                
                loss_total = loss_supervised + lambda_subj * loss_subj
                loss_total.backward()
                optimizer_ft.step()
                
                total_supervised_loss += loss_total.item()
                
            print(f"    Epoch {epoch+1}/{FINE_TUNE_EPOCHS} - Total Supervised Loss: {total_supervised_loss/len(train_loader):.4f}")
            
        # ---------------------------------------------------------
        # STAGE 5: Threshold Validation & Inference
        # ---------------------------------------------------------
        model.eval()
        fold_probs = []
        fold_trues = []
        fold_confs = []
        
        with torch.no_grad():
            for batch in val_loader:
                eye, mouth, gface, prosody, spectral, quality, cardio, motion, label, _ = [x.to(DEVICE) for x in batch]
                
                stress_logits, _, confidence = model(
                    eye, mouth, gface, prosody, spectral, quality, cardio, motion, 
                    return_confidence=True
                )
                probs = torch.softmax(stress_logits, dim=1)[:, 1].cpu().numpy()
                fold_probs.append(probs)
                fold_trues.append(label.cpu().numpy())
                fold_confs.append(confidence.squeeze().cpu().numpy())
                
        fold_probs = np.hstack(fold_probs)
        fold_trues = np.hstack(fold_trues)
        fold_confs = np.hstack(fold_confs)
        
        fold_metrics = calculate_metrics(fold_trues, fold_probs)
        folds_metrics.append(fold_metrics)
        preds_y_true.append(fold_trues)
        preds_y_prob.append(fold_probs)
        preds_y_conf.append(fold_confs)
        
        print(f"  Fold {fold+1} Accuracy: {fold_metrics['Accuracy']:.4f} | F1-Score: {fold_metrics['F1-Score']:.4f} | Mean Confidence: {np.mean(fold_confs):.4f}")
        
    # Compile outputs
    preds_y_true = np.hstack(preds_y_true)
    preds_y_prob = np.hstack(preds_y_prob)
    preds_y_conf = np.hstack(preds_y_conf)
    
    mean_acc = np.mean([f["Accuracy"] for f in folds_metrics])
    mean_f1 = np.mean([f["F1-Score"] for f in folds_metrics])
    mean_auc = np.mean([f["ROC-AUC"] for f in folds_metrics])
    
    print("\n==========================================")
    print("SSVB-CASA-AIS LOSO RESULTS SUMMARY")
    print(f"  Mean Accuracy: {mean_acc:.4f}")
    print(f"  Mean F1-Score: {mean_f1:.4f}")
    print(f"  Mean ROC-AUC:  {mean_auc:.4f}")
    print(f"  Mean Output Confidence: {np.mean(preds_y_conf):.4f}")
    print("==========================================\n")
    
    # Save visual reports
    generate_visual_reports("SSVB-CASA-AIS Model", preds_y_true, preds_y_prob, folds_metrics)
    
    # Export metrics to JSON
    summary_report = {
        "model_name": "SSVB-CASA-AIS (Multi-Head Self-Attention + Contrastive SSL + GRL + Aux Confidence Head)",
        "mean_accuracy": float(mean_acc),
        "mean_f1": float(mean_f1),
        "mean_roc_auc": float(mean_auc),
        "mean_confidence": float(np.mean(preds_y_conf)),
        "folds_metrics": folds_metrics
    }
    
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(summary_report, f, indent=4)
        
    print(f"Reports saved successfully in {REPORTS_DIR}")

def main():
    df_merged = load_and_align_data()
    groups = df_merged['subject_id'].values
    gkf = GroupKFold(n_splits=5)
    train_and_validate_loso(df_merged, groups, gkf)

if __name__ == "__main__":
    main()
