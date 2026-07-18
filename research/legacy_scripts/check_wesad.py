import sys
from ucimlrepo import fetch_ucirepo

try:
    print("Fetching WESAD dataset (id=465)...")
    wesad = fetch_ucirepo(id=465)
    print("Metadata keys:", wesad.keys())
    
    # Check data shapes
    print("Features shape:", wesad.data.features.shape)
    print("Targets shape:", wesad.data.targets.shape)
    
    # Print first few columns and types
    print("\nFeatures columns:")
    print(list(wesad.data.features.columns)[:20])
    
    print("\nTargets columns:")
    print(list(wesad.data.targets.columns))
    
    # Print variables table
    print("\nVariables description:")
    print(wesad.variables)

except Exception as e:
    print("Error:", e)
