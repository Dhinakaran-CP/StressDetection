import pickle
from pathlib import Path

def main():
    pkl_path = Path("data/wesad/S2/S2.pkl")
    if not pkl_path.exists():
        print("S2.pkl not found.")
        return
        
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        # Since it's python3, we need encoding='latin1' or encoding='utf-8' if it was pickled in python2
        data = pickle.load(f, encoding='latin1')
        
    print("\nKeys in S2.pkl:", data.keys())
    print("Subject:", data['subject'])
    
    signal = data['signal']
    print("\nSignal keys:", signal.keys())
    print("Chest keys:", signal['chest'].keys())
    print("Wrist keys:", signal['wrist'].keys())
    
    print("\nChest ECG shape:", signal['chest']['ECG'].shape)
    print("Chest EDA shape:", signal['chest']['EDA'].shape)
    print("Chest ACC shape:", signal['chest']['ACC'].shape)
    
    print("\nWrist BVP shape:", signal['wrist']['BVP'].shape)
    print("Wrist EDA shape:", signal['wrist']['EDA'].shape)
    print("Wrist TEMP shape:", signal['wrist']['TEMP'].shape)
    print("Wrist ACC shape:", signal['wrist']['ACC'].shape)
    
    print("\nLabels shape:", data['label'].shape)
    import numpy as np
    unique_labels, counts = np.unique(data['label'], return_counts=True)
    print("Unique labels and counts:", dict(zip(unique_labels, counts)))

if __name__ == '__main__':
    main()
