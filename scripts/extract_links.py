import re
from pathlib import Path

def main():
    html_path = Path("data/WESAD.zip")
    if not html_path.exists():
        print("HTML page file not found.")
        return
        
    content = html_path.read_text(encoding='latin1')
    
    import base64
    print("=== Found and Decoded Initial State inputs ===")
    matches = re.findall(r'id=["\']initial-state-([^"\']+)["\']\s+value=["\']([^"\']+)["\']', content)
    for key, val in matches:
        try:
            decoded = base64.b64decode(val).decode('utf-8', errors='ignore')
            print(f"{key}: {decoded[:300]}")
        except Exception as e:
            print(f"Error decoding {key}: {e}")


if __name__ == '__main__':
    main()
