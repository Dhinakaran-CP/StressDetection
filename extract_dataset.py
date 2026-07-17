import os
import zipfile
import glob
from pathlib import Path

def main():
    base_dir = Path(r"c:\Users\StressProject\Desktop\StressDetectionUsingML")
    data_dir = base_dir / "data" / "stress_d"
    out_dir = base_dir / "dataset_discovery" / "extracted_preview"
    
    # Create required output directories from dataset_emp.md
    folders = [
        "zip_inventory",
        "extracted_preview",
        "schema_reports",
        "modality_maps",
        "label_maps",
        "timestamp_maps",
        "logs",
        "decision_reports"
    ]
    for folder in folders:
        (base_dir / "dataset_discovery" / folder).mkdir(parents=True, exist_ok=True)

    zip_files = glob.glob(str(data_dir / "*.zip"))
    
    inventory_path = base_dir / "dataset_discovery" / "zip_inventory" / "inventory.txt"
    with open(inventory_path, "w", encoding="utf-8") as f:
        f.write("Filename\tSize(Bytes)\tSubjectID\n")
    
    for zip_path in zip_files:
        path_obj = Path(zip_path)
        filename = path_obj.name
        size = path_obj.stat().st_size
        subject_id = path_obj.stem # e.g. 'S1'
        
        # Append to inventory
        with open(inventory_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{size}\t{subject_id}\n")
            
        # Extract
        target_folder = out_dir / subject_id
        if not target_folder.exists():
            print(f"Extracting {filename} to {target_folder}...")
            target_folder.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)
                print(f"Extraction of {filename} completed.")
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")
        else:
            print(f"Folder {target_folder} already exists, skipping extraction for {filename}.")

if __name__ == '__main__':
    main()
