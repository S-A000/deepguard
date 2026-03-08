import torch
import torch.nn as nn
from transformers import TimesformerModel

class VisualExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(VisualExpert, self).__init__()
        # Facebook ka pre-trained TimeSformer
        self.timesformer = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.fc = nn.Linear(self.timesformer.config.hidden_size, embed_dim)

    def forward(self, video_frames):
        # video_frames shape: (batch, num_frames, channels, height, width)
        outputs = self.timesformer(video_frames)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token)