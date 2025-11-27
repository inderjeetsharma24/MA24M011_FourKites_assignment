"""Model zoo with configurable widths and batch norm."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MLP(nn.Module):
    def __init__(self, width: int, depth: int = 2):
        super().__init__()
        layers = [Flatten()]
        in_dim = 28 * 28
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvNet3(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        channels = [1, base, base * 2, base * 4]
        blocks = []
        for i in range(3):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * (28 // 8) * (28 // 8), 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, hidden_size: int) -> nn.Module:
    n = name.lower()
    if n == "mlp2":
        return MLP(hidden_size, depth=2)
    if n == "mlp3":
        return MLP(hidden_size, depth=3)
    if n == "convnet3":
        return ConvNet3(base=max(16, hidden_size // 32))
    if n == "lenet":
        model = tv_models.lenet.LeNet(num_classes=10)
        return model
    if n == "resnet18":
        model = tv_models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown architecture '{name}'")
