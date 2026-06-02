import torch
import torch.nn as nn
import torchvision.models as models


class ForensicExpert(nn.Module):
    """
    Forensic / FFT Expert Branch

    Supports:
    1. Single-frame FFT:
       input shape = (B, 1, 224, 224)

    2. Multi-frame averaged FFT:
       input shape = (B, 1, 224, 224)
       Recommended because old ResNet-style architecture remains compatible.

    3. Multi-frame stacked FFT:
       input shape = (B, T, 224, 224)
       For this, set in_channels=T.
       Example:
           ForensicExpert(embed_dim=256, in_channels=8)
           ForensicExpert(embed_dim=256, in_channels=16)
    """

    def __init__(
        self,
        embed_dim=256,
        in_channels=1,
        pretrained=True,
        dropout=0.2
    ):
        super(ForensicExpert, self).__init__()

        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # ------------------------------------------
        # ResNet18 backbone
        # ------------------------------------------
        try:
            # New torchvision API
            if pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None

            self.resnet = models.resnet18(weights=weights)

        except Exception:
            # Old torchvision API fallback
            self.resnet = models.resnet18(pretrained=pretrained)

        # ------------------------------------------
        # Replace first conv layer for FFT channels
        # ------------------------------------------
        old_conv = self.resnet.conv1

        self.resnet.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # ------------------------------------------
        # Initialize first conv properly
        # ------------------------------------------
        with torch.no_grad():
            if pretrained and old_conv.weight.shape[1] == 3:
                old_weight = old_conv.weight.data

                if in_channels == 1:
                    # RGB pretrained weights → grayscale FFT
                    self.resnet.conv1.weight.data = old_weight.mean(dim=1, keepdim=True)

                elif in_channels == 3:
                    self.resnet.conv1.weight.data = old_weight

                else:
                    # RGB pretrained weights → multi-frame FFT channels
                    # Mean RGB filter repeated for T channels
                    mean_weight = old_weight.mean(dim=1, keepdim=True)
                    repeated_weight = mean_weight.repeat(1, in_channels, 1, 1)
                    repeated_weight = repeated_weight / in_channels
                    self.resnet.conv1.weight.data = repeated_weight

            else:
                nn.init.kaiming_normal_(
                    self.resnet.conv1.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

        # ------------------------------------------
        # Replace final FC with embedding projection
        # ------------------------------------------
        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, fft_images):
        """
        Expected:
            Single-frame / averaged multi-frame:
                (B, 1, H, W)

            Stacked multi-frame FFT:
                (B, T, H, W)

            If input accidentally comes as:
                (B, H, W), it will be converted to (B, 1, H, W)
        """

        if fft_images.dim() == 3:
            fft_images = fft_images.unsqueeze(1)

        if fft_images.dim() != 4:
            raise ValueError(
                f"Invalid FFT input shape: {fft_images.shape}. "
                "Expected (B, C, H, W)."
            )

        if fft_images.shape[1] != self.in_channels:
            raise ValueError(
                f"FFT channel mismatch. Model expects {self.in_channels} channels, "
                f"but got {fft_images.shape[1]} channels. "
                "If using averaged multi-frame FFT, input should be (B, 1, H, W). "
                "If using stacked FFT, initialize ForensicExpert with in_channels=T."
            )

        fft_images = fft_images.float()
        fft_images = torch.nan_to_num(
            fft_images,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

        embedding = self.resnet(fft_images)

        return embedding
