import torch
import torch.nn as nn

# Experts branches import (Ensure these paths match your folder structure)
from .branch_a_spatial import VisualExpert
from .branch_b_physics import PhysicsExpert
from .branch_c_forensics import ForensicExpert
from .branch_d_audio import AudioExpert

class DeepGuardFusionModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(DeepGuardFusionModel, self).__init__()
        
        # 1. Experts Initialization
        self.visual_expert = VisualExpert(embed_dim=embed_dim)
        self.physics_expert = PhysicsExpert(embed_dim=embed_dim)
        self.forensic_expert = ForensicExpert(embed_dim=embed_dim)
        self.audio_expert = AudioExpert(embed_dim=embed_dim)
        
        # 2. Normalization for training stability
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. Multi-Head Attention (Cross-Modal interaction)
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. Final Decision MLP
        # Hum 4 streams ko average karke 1 vector (embed_dim) par le aaye hain
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Single logit for Binary Classification
        )

    def forward(self, video_frames, optical_flow, fft_images, audio_waveforms):
        # Extract features from each branch
        vis_feat = self.visual_expert(video_frames)
        phys_feat = self.physics_expert(optical_flow)
        for_feat = self.forensic_expert(fft_images)
        aud_feat = self.audio_expert(audio_waveforms)
        
        # Stack into a sequence: (Batch, 4, Embed_Dim)
        stacked_features = torch.stack((vis_feat, phys_feat, for_feat, aud_feat), dim=1)
        
        # Apply LayerNorm before Attention
        stacked_features = self.norm(stacked_features)
        
        # Self-Attention across modalities
        attn_out, _ = self.attention_fusion(stacked_features, stacked_features, stacked_features)
        
        # 🚀 GLOBAL AVERAGE POOLING: Charon modalities ka nichore (mean) nikalna
        # Ye model ko force karta hai ke wo 'Fake' features dhoonde chahe wo kisi bhi branch mein hon
        pooled_features = torch.mean(attn_out, dim=1) 
        
        # Final Classification
        output_logit = self.fusion_mlp(pooled_features)
        
        return output_logit