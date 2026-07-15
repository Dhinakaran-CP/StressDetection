import os
import json
import glob

def find_metrics():
    search_paths = [
        "evaluation_reports/production_model/**/metrics.json",
        "high_capacity_research/early_fusion/reports/**/metrics.json"
    ]
    
    results = []
    for path_pattern in search_paths:
        for filepath in glob.glob(path_pattern, recursive=True):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                # Check structure
                if isinstance(data, list):
                    # It's a list. Let's see if it's a list of dictionaries (folds or direct metrics)
                    if len(data) > 0 and isinstance(data[0], dict):
                        # Calculate mean from list of dicts
                        accs = [d.get("Accuracy", d.get("accuracy", d.get("test_acc", d.get("val_acc", 0)))) for d in data]
                        f1s = [d.get("F1-Score", d.get("f1_score", d.get("f1", d.get("test_f1", 0)))) for d in data]
                        aucs = [d.get("ROC-AUC", d.get("roc_auc", d.get("auc", d.get("test_auc", 0.5)))) for d in data]
                        
                        mean_acc = sum(accs) / len(accs)
                        mean_f1 = sum(f1s) / len(f1s)
                        mean_auc = sum(aucs) / len(aucs)
                        model_name = os.path.basename(os.path.dirname(filepath))
                    else:
                        print(f"Skipping list file with unknown element types: {filepath}")
                        continue
                elif isinstance(data, dict):
                    model_name = data.get("model_name", os.path.basename(os.path.dirname(filepath)))
                    mean_acc = data.get("mean_accuracy", data.get("accuracy", None))
                    if mean_acc is None and "folds_metrics" in data:
                        folds = data["folds_metrics"]
                        if folds:
                            mean_acc = sum(fold.get("Accuracy", fold.get("accuracy", 0)) for fold in folds) / len(folds)
                    
                    mean_f1 = data.get("mean_f1", data.get("f1_score", data.get("f1", None)))
                    if mean_f1 is None and "folds_metrics" in data:
                        folds = data["folds_metrics"]
                        if folds:
                            mean_f1 = sum(fold.get("F1-Score", fold.get("f1_score", fold.get("f1", 0))) for fold in folds) / len(folds)

                    mean_auc = data.get("mean_roc_auc", data.get("roc_auc", data.get("auc", None)))
                    if mean_auc is None and "folds_metrics" in data:
                        folds = data["folds_metrics"]
                        if folds:
                            mean_auc = sum(fold.get("ROC-AUC", fold.get("roc_auc", 0)) for fold in folds) / len(folds)
                else:
                    print(f"Unknown root type {type(data)} in {filepath}")
                    continue
                
                results.append({
                    "path": filepath,
                    "model_name": model_name,
                    "accuracy": mean_acc,
                    "f1": mean_f1,
                    "auc": mean_auc
                })
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
                
    # Sort by accuracy descending
    results = sorted(results, key=lambda x: x["accuracy"] if x["accuracy"] is not None else 0, reverse=True)
    print("\nParsed Model Performance Summary:")
    print("=" * 110)
    print(f"{'Model Directory / Name':<55} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("=" * 110)
    for r in results:
        name = r["model_name"]
        if len(name) > 53:
            name = name[:50] + "..."
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        f1 = f"{r['f1']:.4f}" if r['f1'] is not None else "N/A"
        auc = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{name:<55} | {acc:<10} | {f1:<10} | {auc:<10}")
    print("=" * 110)

if __name__ == "__main__":
    find_metrics()
