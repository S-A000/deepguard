import torch
import torch.nn as nn

# Expert branches import
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
        # 1. Expert Branches
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
        # token 0 = visual
        # token 1 = physics
        # token 2 = forensic
        # token 3 = audio
        self.modality_embedding = nn.Parameter(
            torch.randn(1, self.num_modalities, embed_dim) * 0.02
        )

        # ==========================================
        # 3. Normalization Before Attention
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
        # 5. Normalization After Attention
        # ==========================================
        self.post_attn_norm = nn.LayerNorm(embed_dim)

        # ==========================================
        # 6. Final Fusion Classifier
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
    # Smart State Dict Cleaner
    # ==========================================
    def _clean_state_dict(self, state_dict, for_expert_only=True):
        """
        Handles these checkpoint formats:

        1. Expert-only checkpoint:
            resnet.conv1.weight
            resnet.bn1.weight
            ...

        2. Full branch checkpoint:
            expert.resnet.conv1.weight
            expert.resnet.bn1.weight
            classifier.0.weight
            ...

        3. DataParallel checkpoint:
            module.expert.resnet.conv1.weight
            module.classifier.0.weight

        4. Wrapped checkpoint:
            {"model_state_dict": ...}
            {"state_dict": ...}
        """

        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Remove DataParallel prefix
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

        # If full model checkpoint has expert.* keys,
        # extract only expert part and remove expert. prefix.
        if for_expert_only and any(k.startswith("expert.") for k in state_dict.keys()):
            expert_state = {}

            for k, v in state_dict.items():
                if k.startswith("expert."):
                    new_key = k.replace("expert.", "", 1)
                    expert_state[new_key] = v

            state_dict = expert_state

        # Remove classifier keys if present
        if for_expert_only:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("classifier.")
            }

        return state_dict

    # ==========================================
    # Load Expert Weights
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
        Recommended files:

        Visual:
            visual_FINAL_expert.pth

        Physics:
            physics_FINAL_expert.pth

        Forensic:
            forensic_FINAL.pth
            or forensic_FINAL_expert.pth

        Audio:
            audio_phase2_expert.pth
        """

        if visual_path is not None:
            print(f"🔁 Loading Visual Expert: {visual_path}")
            state = torch.load(visual_path, map_location=map_location)
            state = self._clean_state_dict(state, for_expert_only=True)
            self.visual_expert.load_state_dict(state, strict=strict)
            print("✅ Visual expert loaded")

        if physics_path is not None:
            print(f"🔁 Loading Physics Expert: {physics_path}")
            state = torch.load(physics_path, map_location=map_location)
            state = self._clean_state_dict(state, for_expert_only=True)
            self.physics_expert.load_state_dict(state, strict=strict)
            print("✅ Physics expert loaded")

        if forensic_path is not None:
            print(f"🔁 Loading Forensic Expert: {forensic_path}")
            state = torch.load(forensic_path, map_location=map_location)
            state = self._clean_state_dict(state, for_expert_only=True)
            self.forensic_expert.load_state_dict(state, strict=strict)
            print("✅ Forensic expert loaded")

        if audio_path is not None:
            print(f"🔁 Loading Audio Expert: {audio_path}")
            state = torch.load(audio_path, map_location=map_location)
            state = self._clean_state_dict(state, for_expert_only=True)
            self.audio_expert.load_state_dict(state, strict=strict)
            print("✅ Audio expert loaded")

    # ==========================================
    # Freeze Experts
    # ==========================================
    def freeze_experts(self):
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
    # Unfreeze Experts
    # ==========================================
    def unfreeze_experts(self):
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
    # Forward
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
        # 1. Expert embeddings
        # Each shape: (B, embed_dim)
        # ==========================================
        vis_feat = self.visual_expert(video_frames)
        phys_feat = self.physics_expert(optical_flow)
        for_feat = self.forensic_expert(fft_images)
        aud_feat = self.audio_expert(audio_waveforms)

        # ==========================================
        # 2. Stack modality tokens
        # Shape: (B, 4, embed_dim)
        # ==========================================
        stacked_features = torch.stack(
            [vis_feat, phys_feat, for_feat, aud_feat],
            dim=1
        )

        # ==========================================
        # 3. Add modality identity embeddings
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
        # 7. Average pooling over modalities
        # Shape: (B, embed_dim)
        # ==========================================
        pooled_features = torch.mean(attn_out, dim=1)

        # ==========================================
        # 8. Final binary classification logit
        # Shape: (B, 1)
        # ==========================================
        output_logit = self.fusion_mlp(pooled_features)

        if return_attention:
            return output_logit, attn_weights

        return output_logit
