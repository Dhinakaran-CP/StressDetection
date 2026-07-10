import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.baselines import ModalityEncoder
from src.models.modality_bank import ModalityBank

class ExpertNetwork(nn.Module):
    """
    Sub-network representing a single specialized expert in the mixture.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class FlexiModalMoE(nn.Module):
    """
    Method 5: FlexiModal MoE (Hybrid Robust Fusion).
    Combines independent encoders, latent projection, learned missing-modality bank,
    Laplace gating routers, sparse expert selection, and a final classification head.
    """
    def __init__(self, face_dim=18, voice_dim=12, physio_dim=5, latent_dim=16, 
                 num_experts=4, top_k=2, laplace_scale=0.1, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.laplace_scale = laplace_scale
        
        # Encoders
        self.face_enc = ModalityEncoder(face_dim, latent_dim)
        self.voice_enc = ModalityEncoder(voice_dim, latent_dim)
        self.physio_enc = ModalityEncoder(physio_dim, latent_dim)
        
        # Learned missing modality bank
        self.modality_bank = ModalityBank(latent_dim)
        
        # Expert list
        self.experts = nn.ModuleList([
            ExpertNetwork(3 * latent_dim, 2 * latent_dim) for _ in range(num_experts)
        ])
        
        # Router network (predicts logits for expert selection)
        self.router = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_experts)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, num_classes)
        )
        
    def forward(self, face_x, voice_x, physio_x, face_mask=None, voice_mask=None, physio_mask=None):
        """
        Forward pass.
        Args:
            face_x (Tensor): [batch_size, seq_len, face_dim]
            voice_x (Tensor): [batch_size, seq_len, voice_dim]
            physio_x (Tensor): [batch_size, seq_len, physio_dim]
            face_mask (Tensor): [batch_size] binary mask (1 if present, 0 if missing)
            voice_mask (Tensor): [batch_size] binary mask
            physio_mask (Tensor): [batch_size] binary mask
        """
        batch_size = face_x.size(0)
        device = face_x.device
        
        # Default masks to all ones if not provided
        if face_mask is None:
            face_mask = torch.ones(batch_size, device=device)
        if voice_mask is None:
            voice_mask = torch.ones(batch_size, device=device)
        if physio_mask is None:
            physio_mask = torch.ones(batch_size, device=device)
            
        # 1. Encode modalities
        f_emb = self.face_enc(face_x)   # [batch_size, latent_dim]
        v_emb = self.voice_enc(voice_x)
        p_emb = self.physio_enc(physio_x)
        
        # 2. Apply Missing-Modality Bank substitution
        f_emb, v_emb, p_emb = self.modality_bank(f_emb, v_emb, p_emb, face_mask, voice_mask, physio_mask)
        
        # 3. Form joint representation
        joint_representation = torch.cat([f_emb, v_emb, p_emb], dim=-1) # [batch_size, 3 * latent_dim]
        
        # 4. Gating / Routing with optional Laplace noise
        router_logits = self.router(joint_representation) # [batch_size, num_experts]
        
        if self.training and self.laplace_scale > 0:
            # Generate Laplace noise: U1, U2 ~ Uniform(0,1); noise = scale * ln(U1/U2)
            u1 = torch.rand_like(router_logits)
            u2 = torch.rand_like(router_logits)
            noise = self.laplace_scale * torch.log((u1 + 1e-8) / (u2 + 1e-8))
            gating_logits = router_logits + noise
        else:
            gating_logits = router_logits
            
        # Compute softmax probabilities over experts
        routing_probs = F.softmax(gating_logits, dim=-1) # [batch_size, num_experts]
        
        # Top-K sparse routing selection
        topk_probs, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        # Normalize top-k probabilities to sum to 1
        topk_probs = topk_probs / (torch.sum(topk_probs, dim=-1, keepdim=True) + 1e-8)
        
        # 5. Expert computation and weighting
        expert_outputs = torch.zeros_like(joint_representation) # [batch_size, 3 * latent_dim]
        
        # Compute expert load-balancing metrics for regularization
        # Fraction of samples routed to each expert
        expert_counts = torch.zeros(self.num_experts, device=device)
        for i in range(self.top_k):
            indices = topk_indices[:, i]
            for exp_idx in range(self.num_experts):
                expert_counts[exp_idx] += torch.sum(indices == exp_idx)
                
        # Load balancing loss: entropy of routing frequencies * probabilities
        f_i = expert_counts / (batch_size * self.top_k + 1e-8)
        P_i = torch.mean(routing_probs, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(f_i * P_i)
        
        # Retrieve outputs of top-k experts
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = topk_indices[b, k].item()
                weight = topk_probs[b, k]
                # Run the specific expert network
                exp_out = self.experts[expert_idx](joint_representation[b])
                expert_outputs[b] += weight * exp_out
                
        # 6. Classification Head
        logits = self.classifier(expert_outputs)
        
        return logits, load_balancing_loss
