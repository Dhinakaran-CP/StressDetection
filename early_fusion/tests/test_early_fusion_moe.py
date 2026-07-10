import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Resolve early_fusion directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.split import SubjectSplitter
from src.models.modality_bank import ModalityBank
from src.models.baselines import EarlyFusionClassifier, GatedFusionClassifier, CrossAttentionFusionClassifier
from src.models.fleximodal_moe import FlexiModalMoE
from src.training.trainer import MultimodalTrainer

# Mock Dataset to test training loops
class MockMultimodalDataset(Dataset):
    def __init__(self, num_samples=16, seq_len=5, face_dim=18, voice_dim=12, physio_dim=5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.face_dim = face_dim
        self.voice_dim = voice_dim
        self.physio_dim = physio_dim
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random inputs and labels
        return {
            'face': torch.randn(self.seq_len, self.face_dim),
            'voice': torch.randn(self.seq_len, self.voice_dim),
            'physio': torch.randn(self.seq_len, self.physio_dim),
            'label': torch.randint(0, 2, ()),
            'face_mask': torch.tensor(1.0 if idx % 3 != 0 else 0.0),
            'voice_mask': torch.tensor(1.0 if idx % 4 != 0 else 0.0),
            'physio_mask': torch.tensor(1.0)
        }

def test_subject_splitter():
    splitter = SubjectSplitter(random_seed=42)
    subjects = [f"sub_{i:02d}" for i in range(20)]
    
    splits = splitter.create_splits(subjects, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Assert no subject leakage
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    
    # Assert all subjects allocated
    assert len(train_set | val_set | test_set) == 20

def test_modality_bank_substitutions():
    latent_dim = 16
    batch_size = 4
    seq_len = 5
    
    bank = ModalityBank(latent_dim)
    
    # Inputs
    face_feats = torch.randn(batch_size, seq_len, latent_dim)
    voice_feats = torch.randn(batch_size, seq_len, latent_dim)
    physio_feats = torch.randn(batch_size, seq_len, latent_dim)
    
    # Mask: sample 0 has missing face, sample 1 has missing voice
    face_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
    voice_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    physio_mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    f_out, v_out, p_out = bank(face_feats, voice_feats, physio_feats, face_mask, voice_mask, physio_mask)
    
    # Assert shapes are preserved
    assert f_out.shape == face_feats.shape
    assert v_out.shape == voice_feats.shape
    assert p_out.shape == physio_feats.shape
    
    # Assert missing elements were replaced by parameter values
    # For sample 0, face should equal missing face embedding sequence
    expected_face_placeholder = bank.missing_face_emb.unsqueeze(0).repeat(batch_size, seq_len, 1)
    assert torch.allclose(f_out[0], expected_face_placeholder[0])
    # Sample 1 face should remain identical to input
    assert torch.allclose(f_out[1], face_feats[1])

def test_baseline_classifiers_forward():
    batch_size = 4
    seq_len = 5
    
    # Inputs
    face = torch.randn(batch_size, seq_len, 18)
    voice = torch.randn(batch_size, seq_len, 12)
    physio = torch.randn(batch_size, seq_len, 5)
    
    # Check Early Fusion
    early_model = EarlyFusionClassifier()
    out_early = early_model(face, voice, physio)
    assert out_early.shape == (batch_size, 2)
    
    # Check Gated Fusion
    gated_model = GatedFusionClassifier()
    out_gated = gated_model(face, voice, physio)
    assert out_gated.shape == (batch_size, 2)
    
    # Check Cross-Attention
    cross_model = CrossAttentionFusionClassifier()
    out_cross = cross_model(face, voice, physio)
    assert out_cross.shape == (batch_size, 2)

def test_fleximodal_moe_forward():
    batch_size = 4
    seq_len = 5
    
    face = torch.randn(batch_size, seq_len, 18)
    voice = torch.randn(batch_size, seq_len, 12)
    physio = torch.randn(batch_size, seq_len, 5)
    
    face_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    voice_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    physio_mask = torch.tensor([1.0, 1.0, 1.0, 1.0])
    
    model = FlexiModalMoE(num_experts=3, top_k=2)
    
    # Training forward pass (includes Laplace noise routing)
    model.train()
    logits, lb_loss = model(face, voice, physio, face_mask, voice_mask, physio_mask)
    assert logits.shape == (batch_size, 2)
    assert isinstance(lb_loss, torch.Tensor)
    assert lb_loss.item() >= 0.0
    
    # Eval forward pass
    model.eval()
    logits_eval, _ = model(face, voice, physio, face_mask, voice_mask, physio_mask)
    assert logits_eval.shape == (batch_size, 2)

def test_trainer_fit_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MockMultimodalDataset()
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = FlexiModalMoE(num_experts=2, top_k=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = MultimodalTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path='outputs/checkpoints/test_checkpoint.pt',
        patience=2
    )
    
    # Fit for 2 epochs
    history = trainer.fit(num_epochs=2)
    
    assert "train_loss" in history
    assert "val_f1" in history
    assert len(history["train_loss"]) == 2
    
    # Cleanup checkpoint if written
    if os.path.exists('outputs/checkpoints/test_checkpoint.pt'):
        os.remove('outputs/checkpoints/test_checkpoint.pt')
