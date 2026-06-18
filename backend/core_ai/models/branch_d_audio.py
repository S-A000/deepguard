import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioExpert, self).__init__()

        # Same base model used during audio_phase2 training
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        hidden_size = self.wav2vec.config.hidden_size

        # IMPORTANT:
        # This must match your trained audio checkpoint.
        # Your checkpoint has:
        # projection.0.weight
        # projection.0.bias
        # projection.1.weight
        # projection.1.bias
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, audio_waveforms):
        """
        Expected input shape:
            (batch, audio_length)

        Example:
            (B, 64000) for 4 seconds at 16kHz
        """

        if audio_waveforms.dim() == 3:
            audio_waveforms = audio_waveforms.mean(dim=1)

        if audio_waveforms.dim() != 2:
            raise ValueError(f"Invalid audio input shape: {audio_waveforms.shape}")

        outputs = self.wav2vec(audio_waveforms)

        pooled = outputs.last_hidden_state.mean(dim=1)

        embedding = self.projection(pooled)

        return embedding
