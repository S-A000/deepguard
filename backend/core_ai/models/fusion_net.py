import torch
import torch.nn as nn

# Experts branches import
from .branch_a_spatial import VisualExpert
from .branch_b_physics import PhysicsExpert
from .branch_c_forensics import ForensicExpert
from .branch_d_audio import AudioExpert


class DeepGuardFusionModel(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        freeze_experts=True,
        dropout=0.4
    ):
        super(DeepGuardFusionModel, self).__init__()

        # ==========================================
        # 1. Experts Initialization
        # ==========================================
        self.visual_expert = VisualExpert(embed_dim=embed_dim)
        self.physics_expert = PhysicsExpert(embed_dim=embed_dim)
        self.forensic_expert = ForensicExpert(embed_dim=embed_dim)
        self.audio_expert = AudioExpert(embed_dim=embed_dim)

        self.embed_dim = embed_dim
        self.num_modalities = 4

        # ==========================================
        # 2. Modality Embedding
        # ==========================================
        # Ye model ko batata hai:
        # token 0 = visual
        # token 1 = physics
        # token 2 = forensic
        # token 3 = audio
        self.modality_embedding = nn.Parameter(
            torch.randn(1, self.num_modalities, embed_dim) * 0.02
        )

        # ==========================================
        # 3. Normalization before attention
        # ==========================================
        self.pre_attn_norm = nn.LayerNorm(embed_dim)

        # ==========================================
        # 4. Multi-Head Self-Attention Fusion
        # ==========================================
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # ==========================================
        # 5. Normalization after attention
        # ==========================================
        self.post_attn_norm = nn.LayerNorm(embed_dim)

        # ==========================================
        # 6. Final Fusion MLP
        # ==========================================
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 1)
        )

        # ==========================================
        # 7. Freeze experts initially
        # ==========================================
        if freeze_experts:
            self.freeze_experts()

    # ==========================================
    # Helper: clean state dict
    # ==========================================
    def _clean_state_dict(self, state_dict):
        """
        Handles:
        - raw state_dict
        - checkpoint dict with model_state_dict
        - checkpoint dict with state_dict
        - DataParallel module. prefix
        """

        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
            }

        return state_dict

    # ==========================================
    # Load all expert weights
    # ==========================================
    def load_expert_weights(
        self,
        visual_path=None,
        physics_path=None,
        forensic_path=None,
        audio_path=None,
        map_location="cpu",
        strict=True
    ):
        """
        Fusion ke liye expert-only .pth files load karo.

        Recommended:
        visual_path   = visual_FINAL_expert.pth
        physics_path  = physics_FINAL_expert.pth
        forensic_path = forensic_FINAL_expert.pth
        audio_path    = audio_phase2_expert.pth
        """

        if visual_path is not None:
            print(f"🔁 Loading Visual Expert: {visual_path}")
            state = torch.load(visual_path, map_location=map_location)
            state = self._clean_state_dict(state)
            self.visual_expert.load_state_dict(state, strict=strict)
            print("✅ Visual expert loaded")

        if physics_path is not None:
            print(f"🔁 Loading Physics Expert: {physics_path}")
            state = torch.load(physics_path, map_location=map_location)
            state = self._clean_state_dict(state)
            self.physics_expert.load_state_dict(state, strict=strict)
            print("✅ Physics expert loaded")

        if forensic_path is not None:
            print(f"🔁 Loading Forensic Expert: {forensic_path}")
            state = torch.load(forensic_path, map_location=map_location)
            state = self._clean_state_dict(state)
            self.forensic_expert.load_state_dict(state, strict=strict)
            print("✅ Forensic expert loaded")

        if audio_path is not None:
            print(f"🔁 Loading Audio Expert: {audio_path}")
            state = torch.load(audio_path, map_location=map_location)
            state = self._clean_state_dict(state)
            self.audio_expert.load_state_dict(state, strict=strict)
            print("✅ Audio expert loaded")

    # ==========================================
    # Freeze experts
    # ==========================================
    def freeze_experts(self):
        """
        Experts freeze kar do.
        Sirf fusion attention + fusion MLP train hoga.
        """

        experts = [
            self.visual_expert,
            self.physics_expert,
            self.forensic_expert,
            self.audio_expert
        ]

        for expert in experts:
            for param in expert.parameters():
                param.requires_grad = False

        print("🧊 Experts frozen. Only fusion layers will train.")

    # ==========================================
    # Unfreeze experts
    # ==========================================
    def unfreeze_experts(self):
        """
        Optional fine-tuning ke liye.
        Isko sirf last few epochs mein low LR ke saath use karo.
        """

        experts = [
            self.visual_expert,
            self.physics_expert,
            self.forensic_expert,
            self.audio_expert
        ]

        for expert in experts:
            for param in expert.parameters():
                param.requires_grad = True

        print("🔥 Experts unfrozen. Full end-to-end fine-tuning enabled.")

    # ==========================================
    # Forward pass
    # ==========================================
    def forward(
        self,
        video_frames,
        optical_flow,
        fft_images,
        audio_waveforms,
        return_attention=False
    ):
        # ==========================================
        # 1. Extract features from each expert
        # Expected output from each expert:
        # (Batch, embed_dim)
        # ==========================================
        vis_feat = self.visual_expert(video_frames)
        phys_feat = self.physics_expert(optical_flow)
        for_feat = self.forensic_expert(fft_images)
        aud_feat = self.audio_expert(audio_waveforms)

        # ==========================================
        # 2. Stack modalities
        # Shape: (Batch, 4, Embed_Dim)
        # ==========================================
        stacked_features = torch.stack(
            [vis_feat, phys_feat, for_feat, aud_feat],
            dim=1
        )

        # ==========================================
        # 3. Add modality identity embedding
        # ==========================================
        stacked_features = stacked_features + self.modality_embedding

        # ==========================================
        # 4. Pre-attention normalization
        # ==========================================
        x = self.pre_attn_norm(stacked_features)

        # ==========================================
        # 5. Self-attention across modalities
        # ==========================================
        attn_out, attn_weights = self.attention_fusion(
            x,
            x,
            x,
            need_weights=True
        )

        # ==========================================
        # 6. Residual connection + post norm
        # ==========================================
        attn_out = self.post_attn_norm(attn_out + stacked_features)

        # ==========================================
        # 7. Global average pooling over 4 modalities
        # Shape: (Batch, Embed_Dim)
        # ==========================================
        pooled_features = torch.mean(attn_out, dim=1)

        # ==========================================
        # 8. Final classification
        # Output shape: (Batch, 1)
        # ==========================================
        output_logit = self.fusion_mlp(pooled_features)

        if return_attention:
            return output_logit, attn_weights

        return output_logit
