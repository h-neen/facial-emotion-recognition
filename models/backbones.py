"""
models/backbones.py
====================
Pretrained backbone wrappers.
Both extract spatial feature maps then project to backbone_out_channels
via a 1x1 conv so the outputs can be added element-wise for fusion.
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientBackbone(nn.Module):
    """
    EfficientNetB0 feature extractor.
    Final feature map shape: (B, out_channels, H, W)
    where H = W = image_size // 32 for image_size=192 -> 6x6
    """

    def __init__(self, out_channels: int = 512, pretrained: bool = True):
        super().__init__()
        weights = (models.EfficientNet_B0_Weights.IMAGENET1K_V1
                   if pretrained else None)
        base = models.efficientnet_b0(weights=weights)

        # Remove classifier and adaptive pool — keep only feature extractor
        # EfficientNetB0 features block outputs 1280 channels at H/32 x W/32
        self.features   = base.features    # (B, 1280, H/32, W/32)
        self.pool       = nn.AdaptiveAvgPool2d((6, 6))   # fixed spatial size

        # Project 1280 -> out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(1280, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 1280, H/32, W/32)
        x = self.pool(x)              # (B, 1280, 6, 6)
        x = self.projection(x)        # (B, out_channels, 6, 6)
        return x

    def freeze(self):
        """Freeze all backbone weights (projection layer stays trainable)."""
        for param in self.features.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad_(True)


class InceptionBackbone(nn.Module):
    """
    InceptionV3 feature extractor.
    InceptionV3 expects input >= 75x75. For 192x192 input
    the Mixed_7c layer outputs 2048 channels at 4x4 spatial.
    We project 2048 -> out_channels at a fixed 6x6 spatial size.
    """

    def __init__(self, out_channels: int = 512, pretrained: bool = True):
        super().__init__()
        weights = (models.Inception_V3_Weights.IMAGENET1K_V1
                   if pretrained else None)
        base = models.inception_v3(weights=weights, aux_logits=True)

        # Build feature extractor from InceptionV3 layers
        # We stop at Mixed_7c (last major inception block)
        self.stem = nn.Sequential(
            base.Conv2d_1a_3x3,
            base.Conv2d_2a_3x3,
            base.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base.Conv2d_3b_1x1,
            base.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            base.Mixed_5b, base.Mixed_5c, base.Mixed_5d,
            base.Mixed_6a, base.Mixed_6b, base.Mixed_6c,
            base.Mixed_6d, base.Mixed_6e,
            base.Mixed_7a, base.Mixed_7b, base.Mixed_7c,  # -> 2048 ch
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))

        # Project 2048 -> out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(2048, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # (B, 2048, ~4, ~4) for 192 input
        x = self.pool(x)          # (B, 2048, 6, 6)
        x = self.projection(x)    # (B, out_channels, 6, 6)
        return x

    def freeze(self):
        for param in self.stem.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.stem.parameters():
            param.requires_grad_(True)
