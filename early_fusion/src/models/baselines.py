import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    """
    Standard temporal encoder for time-series feature windows.
    Applies Conv1D, batch normalization, ReLU activation, and a GRU layer.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_dim]
        # Conv1d expects shape: [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        # Return last hidden state: [batch_size, hidden_dim]
        return x[:, -1, :]

class EarlyFusionClassifier(nn.Module):
    """
    Method 1: Early Fusion Baseline.
    Concatenates modality embeddings at the latent level, followed by dense layers.
    """
    def __init__(self, face_dim=18, voice_dim=12, physio_dim=5, latent_dim=16, num_classes=2):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        
        # Concat size: 3 modalities * latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, num_classes)
        )
        
    def forward(self, face_x, voice_x, physio_x):
        f_emb = self.face_enc(face_x)
        v_emb = self.voice_enc(voice_x)
        p_emb = self.physio_enc(physio_x)
        
        fused = torch.cat([f_emb, v_emb, p_emb], dim=-1)
        return self.classifier(fused)

class GatedFusionClassifier(nn.Module):
    """
    Method 2: Gated Fusion.
    Learns dynamic scalar weights for each modality on a per-sample basis.
    """
    def __init__(self, face_dim=18, voice_dim=12, physio_dim=5, latent_dim=16, num_classes=2):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        
        # Gating network to predict weights for each of the 3 modalities
        self.gate_network = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, num_classes)
        )
        
    def forward(self, face_x, voice_x, physio_x):
        f_emb = self.face_enc(face_x)
        v_emb = self.voice_enc(voice_x)
        p_emb = self.physio_enc(physio_x)
        
        # Concatenate to compute gate weights
        concat_emb = torch.cat([f_emb, v_emb, p_emb], dim=-1)
        gate_weights = F.softmax(self.gate_network(concat_emb), dim=-1) # [batch_size, 3]
        
        g_face = gate_weights[:, 0:1]
        g_voice = gate_weights[:, 1:2]
        g_physio = gate_weights[:, 2:3]
        
        # Weighted sum
        fused = g_face * f_emb + g_voice * v_emb + g_physio * p_emb
        return self.classifier(fused)

class CrossAttentionFusionClassifier(nn.Module):
    """
    Method 3: Cross-Attention Fusion.
    Aligns and interacts modality representations using key-query cross-attention.
    """
    def __init__(self, face_dim=18, voice_dim=12, physio_dim=5, latent_dim=16, num_classes=2):
        super().__init__()
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        
        self.latent_dim = latent_dim
        
        # Pairwise Query, Key, Value projections
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, num_classes)
        )
        
    def forward(self, face_x, voice_x, physio_x):
        f_emb = self.face_enc(face_x) # [batch_size, latent_dim]
        v_emb = self.voice_enc(voice_x)
        p_emb = self.physio_enc(physio_x)
        
        # Combine modalities as sequences for multi-head or standard scaled dot-product attention
        # Shape: [batch_size, 3, latent_dim]
        stacked = torch.stack([f_emb, v_emb, p_emb], dim=1)
        
        Q = self.query_proj(stacked) # [batch_size, 3, latent_dim]
        K = self.key_proj(stacked)   # [batch_size, 3, latent_dim]
        V = self.value_proj(stacked) # [batch_size, 3, latent_dim]
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.latent_dim ** 0.5) # [batch_size, 3, 3]
        attention_weights = F.softmax(scores, dim=-1)
        
        attended = torch.bmm(attention_weights, V) # [batch_size, 3, latent_dim]
        
        # Flatten attended representations for final dense head
        fused = attended.view(attended.size(0), -1) # [batch_size, 3 * latent_dim]
        return self.classifier(fused)
