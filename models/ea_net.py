"""
models/ea_net.py
=================
EA-Net: Attention-based Ensemble Network for Facial Emotion Recognition.
Assembles EfficientBackbone + InceptionBackbone + CAM + SAM + Dense head.

Architecture (Paper Section 2.4):
    Input (B, 3, 192, 192)
        |
        +-----> EfficientNetB0 -> 1x1 proj -> (B, 512, 6, 6) --+
        |                                                        |  add
        +-----> InceptionV3    -> 1x1 proj -> (B, 512, 6, 6) --+
                                                                 |
                                              ChannelAttention (Eqs 5-10)
                                                                 |
                                              SpatialAttention  (Eqs 11-15)
                                                                 |
                                              GAP -> Dropout -> FC(7)
"""

import torch
import torch.nn as nn

from models.backbones import EfficientBackbone, InceptionBackbone
from models.attention import ChannelAttention, SpatialAttention


class EANet(nn.Module):

    def __init__(
        self,
        num_classes:      int   = 7,
        backbone_channels: int  = 512,
        cam_reduction:    int   = 16,
        dropout_rate:     float = 0.5,
        pretrained:       bool  = True,
    ):
        super().__init__()

        # ── Dual backbones (parallel) ────────────────────────────────────
        self.efficient = EfficientBackbone(
            out_channels=backbone_channels, pretrained=pretrained)
        self.inception  = InceptionBackbone(
            out_channels=backbone_channels, pretrained=pretrained)

        # ── Attention modules (sequential) ───────────────────────────────
        self.cam = ChannelAttention(
            channels=backbone_channels, reduction=cam_reduction)

        # SAM output has 2 * backbone_channels (because of the concat in Eq. 15)
        self.sam = SpatialAttention()
        sam_out_channels = backbone_channels * 2

        # ── Global average pool + classifier head ────────────────────────
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(sam_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._init_new_weights()

    def _init_new_weights(self):
        """Xavier init for non-pretrained layers."""
        for m in [self.cam, self.sam, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                            nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  normalised input images
        Returns:
            logits : (B, num_classes)  raw (pre-softmax) scores
        """
        # ── Parallel backbone extraction ─────────────────────────────────
        feat_eff = self.efficient(x)   # (B, 512, 6, 6)
        feat_inc = self.inception(x)   # (B, 512, 6, 6)

        # ── Fusion via element-wise addition (Paper Section 2.4) ─────────
        fused = feat_eff + feat_inc    # (B, 512, 6, 6)

        # ── Sequential attention ──────────────────────────────────────────
        Fc = self.cam(fused)           # (B, 512,   6, 6)  — Eqs 5-10
        Fs = self.sam(Fc)              # (B, 1024,  6, 6)  — Eqs 11-15

        # ── Classification head ──────────────────────────────────────────
        out = self.gap(Fs)             # (B, 1024, 1, 1)
        out = self.dropout(out.flatten(1))  # (B, 1024)
        logits = self.classifier(out)  # (B, 7)
        return logits

    def freeze_backbones(self):
        """Stage 1: freeze backbone weights, train attention + head only."""
        self.efficient.freeze()
        self.inception.freeze()
        print("[EANet] Backbones FROZEN — training attention + head only.")

    def unfreeze_backbones(self):
        """Stage 2: unfreeze everything for end-to-end fine-tuning."""
        self.efficient.unfreeze()
        self.inception.unfreeze()
        print("[EANet] Backbones UNFROZEN — full end-to-end fine-tuning.")

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EANet] Total params: {total:,}  Trainable: {trainable:,}")
        return total, trainable


def build_model(cfg=None, freeze_backbone: bool = True) -> EANet:
    """Factory: instantiate and optionally freeze backbones."""
    if cfg is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.config import CFG
        cfg = CFG

    model = EANet(
        num_classes=cfg.num_classes,
        backbone_channels=cfg.backbone_out_channels,
        cam_reduction=cfg.cam_reduction,
        dropout_rate=cfg.dropout_rate,
    )
    if freeze_backbone:
        model.freeze_backbones()
    model.count_parameters()
    return model


# ── quick sanity test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_model(freeze_backbone=True).to(device)
    x = torch.randn(2, 3, 192, 192, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: (2, 7)
    print("EANet forward pass OK.")
