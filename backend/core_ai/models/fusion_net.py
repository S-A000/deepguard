import torch
import torch.nn as nn

# Charon experts ko import karna
from .branch_a_spatial import VisualExpert
from .branch_b_physics import PhysicsExpert
from .branch_c_forensics import ForensicExpert
from .branch_d_audio import AudioExpert

class DeepGuardFusionModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super(DeepGuardFusionModel, self).__init__()
        
        # 1. Charon Experts ko initialize karna
        self.visual_expert = VisualExpert(embed_dim=embed_dim)
        self.physics_expert = PhysicsExpert(embed_dim=embed_dim)
        self.forensic_expert = ForensicExpert(embed_dim=embed_dim)
        self.audio_expert = AudioExpert(embed_dim=embed_dim)
        
        # 👑 2. UPGRADE: Dynamic Cross-Modal Attention (The Intelligent Fusion)
        # Model khud decide karega kis stream (video, audio, physics) ko kitna weight dena hai
        self.attention_fusion = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # 3. LATE FUSION MLP (The Decision Maker)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Overfitting rokne ke liye
            
            nn.Linear(512, 128),
            nn.ReLU(),
            
            nn.Linear(128, 1) # Final Output: 1 score (Real ya Fake)
        )

    def forward(self, video_frames, optical_flow, fft_images, audio_waveforms):
        """
        Ye function charon inputs leta hai, un par Attention apply karta hai,
        aur final probability score nikalta hai.
        """
        # Step 1: Har expert se feature vector nikalwana (Size: batch_size, embed_dim)
        vis_feat = self.visual_expert(video_frames)
        phys_feat = self.physics_expert(optical_flow)
        for_feat = self.forensic_expert(fft_images)
        aud_feat = self.audio_expert(audio_waveforms)
        
        # Step 2: Stack features to form a matrix (Size: batch_size, 4 streams, embed_dim)
        stacked_features = torch.stack((vis_feat, phys_feat, for_feat, aud_feat), dim=1)
        
        # Step 3: Apply Dynamic Cross-Attention 🧠
        # Yahan charon streams aapas mein baat karti hain (Self-Attention)
        attended_features, attention_weights = self.attention_fusion(
            stacked_features, stacked_features, stacked_features
        )
        
        # Step 4: Flatten the attended features for the final MLP
        # Size ho jayega: (batch_size, 4 * embed_dim)
        flattened_features = attended_features.reshape(attended_features.size(0), -1)
        
        # Step 5: Final Faisla (MLP)
        output_logit = self.fusion_mlp(flattened_features)
        
        return output_logit