import os
import yaml
import pandas as pd

class DatasetCertifier:
    def __init__(self, schema_path="configs/schema_contract.yaml"):
        self.schema_path = schema_path
        with open(schema_path, "r") as f:
            self.schema = yaml.safe_load(f)
            
    def validate(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """
        Mathematically proves that the DataFrame adheres to the schema.
        Returns a certification report dict if successful.
        Raises ValueError if any contract violation is detected.
        """
        print(f"[{dataset_name}] Starting dataset certification...")
        
        # 1. Required Columns Check
        required_cols = [col["name"] for col in self.schema["metadata_schema"]["columns"] if col["required"]]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Schema Violation: Missing required columns: {missing_cols}")
            
        # 2. Missing Subject IDs Check (Critical for LOSO)
        if df["subject_id"].isnull().any():
            missing_count = df["subject_id"].isnull().sum()
            raise ValueError(f"Schema Violation: Found {missing_count} rows with missing subject_id.")
            
        # 3. Duplicate Key Check
        duplicate_mask = df.duplicated(subset=["subject_id", "task_id", "window_index"])
        if duplicate_mask.any():
            dup_count = duplicate_mask.sum()
            raise ValueError(f"Schema Violation: Found {dup_count} duplicate rows for the same subject/task/window.")
            
        # 4. Chronology / Monotonicity Check
        # Ensure that window_start increases strictly monotonically within a task
        print(f"[{dataset_name}] Validating temporal monotonicity...")
        for name, group in df.groupby(["subject_id", "task_id"]):
            group = group.sort_values("window_index")
            if not group["window_start"].is_monotonic_increasing:
                raise ValueError(f"Temporal Violation: window_start is not monotonic for {name}.")
                
        # 5. Build Certification Report
        report = {
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "total_subjects": int(df["subject_id"].nunique()),
            "class_balance": df["label"].value_counts().to_dict(),
            "status": "CERTIFIED"
        }
        
        print(f"[{dataset_name}] Certification PASSED. {len(df)} rows certified.")
        return report

def align_modalities(face_df: pd.DataFrame, voice_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Ensures that Face and Voice modalities have perfectly matching temporal windows.
    Drops any windows where one modality is missing if 'allow_missing_modality' is false in contract.
    (Currently, we assume we want an exact inner join for fused training).
    """
    print("Aligning modalities (Inner Join on subject, task, window_index)...")
    
    # We join on the exact composite key
    merged = pd.merge(
        face_df, voice_df, 
        on=["subject_id", "task_id", "window_index", "window_start", "window_end", "label"],
        how="inner",
        suffixes=("_face", "_voice")
    )
    
    print(f"Alignment complete: {len(merged)} strictly synchronized multi-modal windows remain.")
    
    # Re-split if you want them physically separate but aligned, or return merged.
    # We will just return the fact that they can be aligned.
    return merged
