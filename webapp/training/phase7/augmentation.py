import numpy as np
import torch

def apply_jitter(seq, std=0.05):
    """Adds random Gaussian noise to the sequence."""
    noise = np.random.normal(0, std, seq.shape)
    return seq + noise

def apply_scaling(seq, min_scale=0.9, max_scale=1.1):
    """Multiplies sequence by a random scaling factor."""
    scale = np.random.uniform(min_scale, max_scale)
    return seq * scale

def apply_time_mask(seq, mask_ratio=0.2):
    """Zeroes out random time steps in the sequence."""
    seq_len = seq.shape[0]
    num_to_mask = int(seq_len * mask_ratio)
    if num_to_mask > 0:
        mask_indices = np.random.choice(seq_len, num_to_mask, replace=False)
        seq_masked = seq.copy()
        seq_masked[mask_indices] = 0
        return seq_masked
    return seq

def apply_modality_dropout(face_seq, physio_seq, dropout_prob=0.15):
    """Randomly drops (zeroes out) one of the modalities with a given probability."""
    if np.random.rand() < dropout_prob:
        # Choose which modality to drop
        if np.random.rand() < 0.5:
            # Drop face
            return np.zeros_like(face_seq), physio_seq
        else:
            # Drop physio
            return face_seq, np.zeros_like(physio_seq)
    return face_seq, physio_seq
