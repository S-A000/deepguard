import torch
import torch.nn as nn
from transformers import TimesformerModel


class VisualExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(VisualExpert, self).__init__()

        # ✅ Facebook pre-trained TimeSformer
        # IMPORTANT:
        # torch_dtype=torch.float16 remove kar diya hai.
        # Model FP32 mein load hoga, mixed precision training autocast handle karega.
        self.timesformer = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            low_cpu_mem_usage=True
        )

        self.fc = nn.Linear(
            self.timesformer.config.hidden_size,
            embed_dim
        )

    def forward(self, video_frames):
        """
        video_frames expected shape:
        (batch, num_frames, channels, height, width)

        Example:
        (B, 8, 3, 224, 224) or (B, 16, 3, 224, 224)
        """

        # ✅ Correct HuggingFace argument name
        outputs = self.timesformer(pixel_values=video_frames)

        cls_token = outputs.last_hidden_state[:, 0, :]

        embedding = self.fc(cls_token)

        return embedding
