"""Compact CNN for face-manipulation artefacts (CPU-friendly, ~0.67M params).

Input: 128x128 RGB context crop. A fixed (non-trainable) high-pass layer adds
per-channel Laplacian residuals — blending seams, resampling and noise-level
inconsistencies live in the high frequencies that RGB-trained nets tend to
ignore early in training. Plain strided 3x3 convs: fastest to train on CPU.
"""

from __future__ import annotations

import torch
from torch import nn

MEAN, STD = 0.5, 0.25


class HighPass(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        self.register_buffer("weight", k.expand(3, 1, 3, 3).clone())

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, padding=1, groups=3) * 4.0


def conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DeepfakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hp = HighPass()
        self.features = nn.Sequential(
            conv_bn(6, 32, 2),  # 64
            conv_bn(32, 48, 2),  # 32
            conv_bn(48, 64, 1),
            conv_bn(64, 96, 2),  # 16
            conv_bn(96, 128, 1),
            conv_bn(128, 160, 2),  # 8
            conv_bn(160, 192, 2),  # 4
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, 1))

    def forward(self, x):
        x = torch.cat([x, self.hp(x)], dim=1)
        return self.head(self.features(x))
