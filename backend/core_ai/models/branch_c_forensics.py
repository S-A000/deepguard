import torch
import torch.nn as nn
import torchvision.models as models

class ForensicExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(ForensicExpert, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # 1-channel Grayscale (FFT Magnitude) ke liye layer change ki hai
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)

    def forward(self, fft_images):
        return self.resnet(fft_images)