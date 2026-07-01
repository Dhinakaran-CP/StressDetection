import os
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from backend.core.artifact_manifest import ArtifactManifest
from backend.core.version_registry import VersionRegistry

def package_and_release_expert(modality, model, scaler, X_test, y_test, version="1.0.0"):
    """
    Evaluates the trained expert model on the test set, computes metrics,
    pickles the artifacts, creates a manifest, and registers it.
    """
    RELEASE_DIR = "backend/expert_models"
    os.makedirs(RELEASE_DIR, exist_ok=True)
    
    print(f"\n[{modality.upper()} EXPERT] Packaging Release...")
    
    # 1. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"Accuracy: {acc:.4f}")
    
    # 2. Save Model & Scaler
    model_path = os.path.join(RELEASE_DIR, f"{modality}_expert_lightweight.pkl")
    scaler_path = os.path.join(RELEASE_DIR, f"{modality}_scaler_lightweight.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    if scaler:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
    # 3. Create Manifest Metadata
    metadata = {
        "accuracy": float(acc),
        "f1_score_stress": float(report["1"]["f1-score"]) if "1" in report else 0.0,
        "confusion_matrix": cm,
        "evaluation_protocol": "Leave-One-Subject-Out (GroupKFold)",
    }
    
    manifest = ArtifactManifest(f"{modality}_expert_v1", "model", version, metadata=metadata)
    manifest.compute_hash(model_path)
    manifest.save(model_path)
    
    # 4. Register
    registry = VersionRegistry()
    registry.register_model(f"{modality}_expert", manifest)
    print(f"[{modality.upper()} EXPERT] Released successfully with hash {manifest.hash}")
    return manifest
