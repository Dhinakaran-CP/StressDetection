import os
import json
from pathlib import Path
from pipeline.common.io_utils import read_json, write_json

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    metrics_path = base_dir / "pipeline" / "logs" / "model_zoo_metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Model zoo metrics missing at {metrics_path}")
        
    data = read_json(metrics_path)
    
    results = {}
    eligible_models = []
    
    for model_name in ["logistic_regression", "lightgbm", "mlp", "temporal"]:
        sid_auc = data["datasets"]["stressid"]["summary"][model_name]["roc_auc"]
        es_auc = data["datasets"]["empathicschool"]["summary"][model_name]["roc_auc"]
        
        g2_passed = sid_auc > 0.70
        g3_passed = es_auc > 0.55
        
        passed_both = g2_passed and g3_passed
        
        results[model_name] = {
            "stressid_auc": sid_auc,
            "empathicschool_auc": es_auc,
            "gate_g2_passed": bool(g2_passed),
            "gate_g3_passed": bool(g3_passed),
            "eligible_for_production": bool(passed_both)
        }
        
        if passed_both:
            eligible_models.append(model_name)
            
    # Find the best production candidate (highest average AUC across both datasets)
    best_avg_auc = -1.0
    production_candidate = None
    for model_name in eligible_models:
        avg_auc = (results[model_name]["stressid_auc"] + results[model_name]["empathicschool_auc"]) / 2.0
        if avg_auc > best_avg_auc:
            best_avg_auc = avg_auc
            production_candidate = model_name
            
    output = {
        "gates": results,
        "production_eligible_models": eligible_models,
        "selected_production_candidate": production_candidate,
        "gate_g2_threshold": 0.70,
        "gate_g3_threshold": 0.55
    }
    
    write_json(output, base_dir / "pipeline" / "logs" / "generalization_gates.json")
    
    print("\n=== Generalization Gates Verification ===")
    for model_name, res in results.items():
        status = "ELIGIBLE" if res["eligible_for_production"] else "FAILED"
        print(f"Model: {model_name:<20} | StressID AUC: {res['stressid_auc']:.4f} (Gate G2: {'PASS' if res['gate_g2_passed'] else 'FAIL'}) | EmpathicSchool AUC: {res['empathicschool_auc']:.4f} (Gate G3: {'PASS' if res['gate_g3_passed'] else 'FAIL'}) | Status: {status}")
        
    print(f"\nProduction candidate: {production_candidate} (Avg AUC: {best_avg_auc:.4f})")

if __name__ == "__main__":
    main()
