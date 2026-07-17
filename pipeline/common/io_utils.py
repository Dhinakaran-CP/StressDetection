import json
import pandas as pd
from pathlib import Path

def read_csv_or_xls(path, header=None, **kwargs):
    """
    Reads a file as CSV, falling back to Excel if it contains binary OLE headers.
    """
    path_obj = Path(path)
    try:
        # Check magic number first to avoid parser warnings/errors
        with open(path_obj, "rb") as f:
            header_bytes = f.read(8)
        if header_bytes == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return pd.read_excel(path_obj, header=header, **kwargs)
        else:
            return pd.read_csv(path_obj, header=header, **kwargs)
    except Exception:
        # Fallback to excel if reading as csv fails
        try:
            return pd.read_excel(path_obj, header=header, **kwargs)
        except Exception as e:
            raise IOError(f"Failed to read file {path} as CSV or Excel: {e}")

def write_json(data, path):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
