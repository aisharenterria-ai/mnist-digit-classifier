"""ResNet ligera adaptada a MNIST (1 canal, 28x28)."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Bloque residual basico de dos convoluciones."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNetMNIST(nn.Module):
    """ResNet-20 reducido, entrenado desde cero para MNIST (~270k parametros)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1, num_blocks: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser >= 2")
        if num_blocks < 1:
            raise ValueError("num_blocks debe ser >= 1")

        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)
