"""
models/attention.py
====================
Channel Attention Module (CAM)  — Paper Equations 5-10
Spatial Attention Module (SAM)  — Paper Equations 11-15

Both modules are implemented to match the paper's mathematical formulation
exactly, not a generic CBAM variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (C_AM) — Paper Section 2.3.1, Equations 5-10.

    Forward pass summary:
        alpha_avg = AvgPool(alpha)               [Eq. 5]
        alpha_max = MaxPool(alpha)               [Eq. 6]
        Mmax = A(fc2(A(fc1(alpha_max))))         [Eq. 7]  (shared MLP)
        Mavg = A(fc2(A(fc1(alpha_avg))))         [Eq. 8]  (same shared MLP)
        Mc   = Mmax + Mavg                       [Eq. 9]
        Fc   = sigmoid(Mc) * alpha               [Eq. 10] (skip-connection multiply)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels  : number of input channels (= backbone_out_channels)
            reduction : MLP bottleneck ratio (16 as per CBAM paper the
                        EA-Net attention is derived from)
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    # Eq. 5
        self.max_pool = nn.AdaptiveMaxPool2d(1)    # Eq. 6

        # Shared FC layers (Eqs. 7 and 8 use the SAME weights)
        mid = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alpha : (B, C, H, W)  — fused backbone feature map
        Returns:
            Fc    : (B, C, H, W)  — channel-attended feature map
        """
        # Eqs. 5 & 8: average-pool branch
        avg_squeezed = self.avg_pool(alpha)            # (B, C, 1, 1)
        Mavg = self.shared_mlp(avg_squeezed)           # (B, C)

        # Eqs. 6 & 7: max-pool branch
        max_squeezed = self.max_pool(alpha)            # (B, C, 1, 1)
        Mmax = self.shared_mlp(max_squeezed)           # (B, C)

        # Eq. 9: element-wise add
        Mc = Mavg + Mmax                               # (B, C)

        # Eq. 10: sigmoid + skip-connection multiply (broadcast over H, W)
        Mc = self.sigmoid(Mc).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        Fc = Mc * alpha                                # (B, C, H, W)
        return Fc


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (S_AM) — Paper Section 2.3.2, Equations 11-15.

    Forward pass summary:
        alpha_avg = AvgPool_channel(Fc)              [Eq. 11]
        alpha_max = MaxPool_channel(Fc)              [Eq. 12]
        MSFc = A(f1x1(f3x3(A(f1x1(cat(avg,max)))))) [Eq. 13]
        MSFcGAP = GAP(Fc)                           [Eq. 14]
        Fs = concat[MSFcGAP_expanded, Fc]           [Eq. 15]
    """

    def __init__(self):
        super().__init__()
        # Eq. 13: three conv layers with kernel sizes 1x1, 3x3, 1x1
        # Input channels = 2 (avg-pool + max-pool concatenated along channel axis)
        self.conv1 = nn.Conv2d(2,  64, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64,  1, kernel_size=1, padding=0, bias=False)
        self.relu  = nn.ReLU(inplace=True)
        self.gap   = nn.AdaptiveAvgPool2d(1)    # Eq. 14

    def forward(self, Fc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Fc : (B, C, H, W)  — CAM output
        Returns:
            Fs : (B, C+C, H, W)  — spatial-attended feature map
                 (channel dim doubles from concat of GAP(Fc) + Fc)
        """
        # Eqs. 11-12: pool along channel dimension
        avg = Fc.mean(dim=1, keepdim=True)               # (B, 1, H, W)
        mx  = Fc.max(dim=1, keepdim=True).values         # (B, 1, H, W)

        # Concat -> 2-channel spatial descriptor
        cat = torch.cat([avg, mx], dim=1)                # (B, 2, H, W)

        # Eq. 13: f_1x1 -> ReLU -> f_3x3 -> ReLU -> f_1x1
        MSFc = self.relu(self.conv1(cat))                # (B, 64, H, W)
        MSFc = self.relu(self.conv2(MSFc))               # (B, 64, H, W)
        MSFc = self.conv3(MSFc)                          # (B,  1, H, W)

        # Eq. 14: global average pool of the CAM output Fc
        MSFcGAP = self.gap(Fc)                           # (B, C, 1, 1)

        # Eq. 15: expand GAP to spatial size, then concatenate with Fc
        gap_expanded = MSFcGAP.expand_as(Fc)             # (B, C, H, W)
        Fs = torch.cat([gap_expanded, Fc], dim=1)        # (B, 2C, H, W)
        return Fs
