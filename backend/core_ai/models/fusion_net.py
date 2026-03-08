import torch
import torch.nn as nn

# Charon experts ko import karna (jo files aapne abhi banayi hain)
from .branch_a_spatial import VisualExpert
from .branch_b_physics import PhysicsExpert
from .branch_c_forensics import ForensicExpert
from .branch_d_audio import AudioExpert

class DeepGuardFusionModel(nn.Module):
    def __init__(self, embed_dim=256):
        super(DeepGuardFusionModel, self).__init__()
        
        # 1. Charon Experts ko initialize karna
        self.visual_expert = VisualExpert(embed_dim=embed_dim)
        self.physics_expert = PhysicsExpert(embed_dim=embed_dim)
        self.forensic_expert = ForensicExpert(embed_dim=embed_dim)
        self.audio_expert = AudioExpert(embed_dim=embed_dim)
        
        # 2. LATE FUSION MLP (The Decision Maker)
        # Charon models ka data mil kar (256 * 4 = 1024) features banayega
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
        Ye function charon inputs ek sath leta hai aur final result deta hai.
        """
        # Step 1: Har expert se feature vector nikalwana
        vis_feat = self.visual_expert(video_frames)
        phys_feat = self.physics_expert(optical_flow)
        for_feat = self.forensic_expert(fft_images)
        aud_feat = self.audio_expert(audio_waveforms)
        
        # Step 2: Sab ko jorna (Concatenation / Late Fusion)
        # Dim 1 par jor rahe hain (Batch size same rahega)
        fused_features = torch.cat((vis_feat, phys_feat, for_feat, aud_feat), dim=1)
        
        # Step 3: Final Faisla (MLP)
        output_logit = self.fusion_mlp(fused_features)
        
        return output_logit