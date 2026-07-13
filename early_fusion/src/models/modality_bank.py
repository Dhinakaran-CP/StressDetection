import torch
import torch.nn as nn

class ModalityBank(nn.Module):
    """
    Implements a bank of trainable parameter embeddings representing missing inputs
    for Face, Voice, and Physio modalities.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Trainable parameter embeddings representing a missing window
        self.missing_face_emb = nn.Parameter(torch.randn(1, latent_dim))
        self.missing_voice_emb = nn.Parameter(torch.randn(1, latent_dim))
        self.missing_physio_emb = nn.Parameter(torch.randn(1, latent_dim))
        
        # Initialize with standard normal distributions scaled by projection size
        nn.init.normal_(self.missing_face_emb, std=0.02)
        nn.init.normal_(self.missing_voice_emb, std=0.02)
        nn.init.normal_(self.missing_physio_emb, std=0.02)
        
    def forward(self, face_feats, voice_feats, physio_feats, face_mask, voice_mask, physio_mask):
        """
        Replaces missing modality vectors with the learned parameters.
        Args:
            face_feats (Tensor): [batch_size, seq_len, latent_dim] or [batch_size, latent_dim]
            voice_feats (Tensor): Same shape as face_feats
            physio_feats (Tensor): Same shape as face_feats
            face_mask (Tensor): Binary mask [batch_size] (True if Face exists, False if missing)
            voice_mask (Tensor): Binary mask [batch_size]
            physio_mask (Tensor): Binary mask [batch_size]
        """
        batch_size = face_feats.size(0)
        has_seq = len(face_feats.shape) == 3
        
        if has_seq:
            seq_len = face_feats.size(1)
            # Expand parameter embeddings to [batch_size, seq_len, latent_dim]
            face_placeholder = self.missing_face_emb.unsqueeze(0).repeat(batch_size, seq_len, 1)
            voice_placeholder = self.missing_voice_emb.unsqueeze(0).repeat(batch_size, seq_len, 1)
            physio_placeholder = self.missing_physio_emb.unsqueeze(0).repeat(batch_size, seq_len, 1)
            
            # Format mask shape to align for broadcasting: [batch_size, 1, 1]
            f_mask = face_mask.view(batch_size, 1, 1).float()
            v_mask = voice_mask.view(batch_size, 1, 1).float()
            p_mask = physio_mask.view(batch_size, 1, 1).float()
        else:
            face_placeholder = self.missing_face_emb.repeat(batch_size, 1)
            voice_placeholder = self.missing_voice_emb.repeat(batch_size, 1)
            physio_placeholder = self.missing_physio_emb.repeat(batch_size, 1)
            
            f_mask = face_mask.view(batch_size, 1).float()
            v_mask = voice_mask.view(batch_size, 1).float()
            p_mask = physio_mask.view(batch_size, 1).float()
            
        # Reconstruct representation by dynamic interpolation/masking
        out_face = face_feats * f_mask + face_placeholder * (1.0 - f_mask)
        out_voice = voice_feats * v_mask + voice_placeholder * (1.0 - v_mask)
        out_physio = physio_feats * p_mask + physio_placeholder * (1.0 - p_mask)
        
        return out_face, out_voice, out_physio
