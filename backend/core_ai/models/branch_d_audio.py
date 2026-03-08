import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(AudioExpert, self).__init__()
        # Facebook ka pre-trained Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.fc = nn.Linear(self.wav2vec.config.hidden_size, embed_dim)

    def forward(self, audio_waveforms):
        outputs = self.wav2vec(audio_waveforms)
        # Audio sequence ko pool karke single feature vector banana
        pooled_audio = torch.mean(outputs.last_hidden_state, dim=1)
        return self.fc(pooled_audio)