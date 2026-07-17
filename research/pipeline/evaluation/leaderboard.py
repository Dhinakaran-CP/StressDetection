import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.common.io_utils import read_json

def main():
    base_dir = Path(__file__).resolve().parents[3]
    metrics_path = base_dir / "pipeline" / "logs" / "model_zoo_metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Model zoo metrics missing at {metrics_path}")
        
    data = read_json(metrics_path)
    
    md = []
    md.append("# Master Leaderboard\n")
    md.append("This document summarizes the Leave-One-Subject-Out (LOSO) cross-validation performance of all trained models in the Model Zoo. Models are ranked by F1 stability (F1-macro mean - standard deviation).\n")
    
    for ds_name in ["stressid", "empathicschool", "combined"]:
        ds_title = "StressID" if ds_name == "stressid" else ("EmpathicSchool" if ds_name == "empathicschool" else "Combined 95-Subject")
        md.append(f"## {ds_title} Dataset Leaderboard\n")
        
        summary = data["datasets"][ds_name]["summary"]
        fold_details = data["datasets"][ds_name]["fold_details"]
        
        rows = []
        for model_name in summary.keys():
            # Calculate standard deviation of F1 across folds
            f1_scores = [fold["f1"] for fold in fold_details[model_name]]
            f1_std = np.std(f1_scores) if len(f1_scores) > 0 else 0.0
            
            rows.append({
                "Model": model_name,
                "Accuracy": summary[model_name]["accuracy"],
                "Balanced Accuracy": summary[model_name]["balanced_accuracy"],
                "Precision": summary[model_name]["precision"],
                "Recall": summary[model_name]["recall"],
                "F1": summary[model_name]["f1"],
                "ROC-AUC": summary[model_name]["roc_auc"],
                "PR-AUC": summary[model_name]["pr_auc"],
                "F1 Std Dev": f1_std,
                "Ranking Score": summary[model_name]["f1"] - f1_std
            })
            
        # Find best model based on F1-Score - F1 Std Dev stability ranking
        best_model = max(rows, key=lambda x: x["Ranking Score"])
        
        md.append("| Model Archetype | Accuracy | Bal Acc | Precision | Recall | F1-Score | AUC-ROC | PR-AUC | F1 Std Dev | Notes |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in rows:
            is_best = "⭐ **Top Performer (Stable)**" if r["Model"] == best_model["Model"] else ""
            md.append(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Balanced Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC-AUC']:.4f} | {r['PR-AUC']:.4f} | {r['F1 Std Dev']:.4f} | {is_best} |")
        md.append("\n")
        
    md_content = "\n".join(md)
    
    # Save to local workspace
    local_path = base_dir / "pipeline" / "logs" / "master_leaderboard.md"
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    # Save to appData artifact directory
    artifact_dir = Path(r"C:\Users\StressProject.DESKTOP-U6P7JQT\.gemini\antigravity-ide\brain\d5b9e69d-6cdc-46e6-91bf-81527fec7dfc")
    if artifact_dir.exists():
        with open(artifact_dir / "master_leaderboard.md", "w", encoding="utf-8") as f:
            f.write(md_content)
            
    print("Master leaderboard generated successfully.")

if __name__ == "__main__":
    main()
