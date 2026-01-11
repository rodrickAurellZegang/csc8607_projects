"""
Construction du modèle CNN pour SVHN.

Architecture :
- 3 stages
- Blocs Conv3x3 -> BatchNorm -> ReLU
- Downsampling par stride (pas de MaxPool)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Bloc de base : Conv3x3 -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CNN_SVHN(nn.Module):
    def __init__(self, channels, num_classes, extra_block):
        super().__init__()

        c1, c2, c3 = channels

        # -------- Stage 1 --------
        blocks_stage1 = [
            ConvBlock(3, c1, stride=1),
            ConvBlock(c1, c1, stride=1),
        ]
        if extra_block:
            blocks_stage1.append(ConvBlock(c1, c1, stride=1))

        self.stage1 = nn.Sequential(*blocks_stage1)

        # -------- Stage 2 --------
        blocks_stage2 = [
            ConvBlock(c1, c2, stride=2),  # downsampling
            ConvBlock(c2, c2, stride=1),
        ]
        if extra_block:
            blocks_stage2.append(ConvBlock(c2, c2, stride=1))

        self.stage2 = nn.Sequential(*blocks_stage2)

        # -------- Stage 3 --------
        blocks_stage3 = [
            ConvBlock(c2, c3, stride=2),  # downsampling
            ConvBlock(c3, c3, stride=1),
        ]
        if extra_block:
            blocks_stage3.append(ConvBlock(c3, c3, stride=1))

        self.stage3 = nn.Sequential(*blocks_stage3)

        # Global Average Pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.gap(x)              # (B, C, 1, 1)
        x = torch.flatten(x, 1)      # (B, C)
        x = self.classifier(x)       # (B, num_classes)

        return x


def build_model(config: dict) -> nn.Module:
    """
    Construit et retourne le modèle CNN selon la config.
    """

    model = CNN_SVHN(
        channels=config["model"]["channels"],
        num_classes=config["model"]["num_classes"],
        extra_block=config["model"]["extra_block"],
    )

    return model
