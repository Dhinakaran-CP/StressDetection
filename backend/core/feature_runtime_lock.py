import yaml
import numpy as np

class FeatureRuntimeLock:
    """
    Guarantees that offline training and live webcam inference use the exact
    same mathematical transformations, feature dimensions, and null handling.
    """
    def __init__(self, contract_path="configs/feature_contract.yaml"):
        with open(contract_path, "r") as f:
            self.contract = yaml.safe_load(f)
            
    def process_face_features(self, raw_features, scaler=None):
        """
        Locks face features to exactly 18 dimensions, applies zero-fill for missing,
        and scales using the trained StandardScaler.
        """
        target_dim = self.contract["modalities"]["face"]["input_shape"]
        
        if raw_features is None:
            raw_features = np.zeros(target_dim)
            
        features = np.array(raw_features, dtype=np.float64)
        
        # Missing value strategy
        if self.contract["modalities"]["face"]["missing_value_strategy"] == "zero_fill":
            features = np.nan_to_num(features, nan=0.0)
            
        # Assert dimensionality
        if features.shape[0] != target_dim:
            raise ValueError(f"FeatureRuntimeLock Violation: Face expected {target_dim} dims, got {features.shape[0]}")
            
        features = features.reshape(1, -1)
        
        if scaler is not None:
            features = scaler.transform(features)
            
        return features

    def process_voice_features(self, raw_features, scaler=None):
        """
        Locks voice features to exactly 12 dimensions, applies zero-fill for missing,
        and scales using the trained StandardScaler.
        """
        target_dim = self.contract["modalities"]["voice"]["input_shape"]
        
        if raw_features is None:
            raw_features = np.zeros(target_dim)
            
        features = np.array(raw_features, dtype=np.float64)
        
        # Missing value strategy
        if self.contract["modalities"]["voice"]["missing_value_strategy"] == "zero_fill":
            features = np.nan_to_num(features, nan=0.0)
            
        # Assert dimensionality
        if features.shape[0] != target_dim:
            raise ValueError(f"FeatureRuntimeLock Violation: Voice expected {target_dim} dims, got {features.shape[0]}")
            
        features = features.reshape(1, -1)
        
        if scaler is not None:
            features = scaler.transform(features)
            
        return features

    def process_physio_features(self, raw_features, scaler=None):
        """
        Locks physio features to exactly the specified dimensions, applies the
        contract-specified missing value strategy (zero_fill or mean_fill),
        and scales using the trained StandardScaler.
        """
        target_dim = self.contract["modalities"]["physio"]["input_shape"]
        
        if raw_features is None:
            raw_features = np.zeros(target_dim)
            
        features = np.array(raw_features, dtype=np.float64)
        
        # Apply missing value strategy from contract
        strategy = self.contract["modalities"]["physio"].get("missing_value_strategy", "zero_fill")
        if strategy == "zero_fill":
            features = np.nan_to_num(features, nan=0.0)
        elif strategy == "mean_fill":
            # Replace NaN with column mean; if all-NaN fall back to zero
            nan_mask = np.isnan(features)
            if nan_mask.any():
                valid_mean = np.nanmean(features) if not np.all(nan_mask) else 0.0
                features[nan_mask] = valid_mean
        else:
            # Safety net: always clear NaN regardless of strategy
            features = np.nan_to_num(features, nan=0.0)
            
        # Assert dimensionality
        if features.shape[0] != target_dim:
            raise ValueError(f"FeatureRuntimeLock Violation: Physio expected {target_dim} dims, got {features.shape[0]}")
            
        features = features.reshape(1, -1)
        
        if scaler is not None:
            features = scaler.transform(features)
            
        return features
