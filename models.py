"""
Neural Network Model Architectures
"""

import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten layer for converting conv outputs to linear layers"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class MLP2(nn.Module):
    """2-layer Multi-Layer Perceptron - Wide Network for Benign Regime"""
    def __init__(self, hidden_size: int = 4096, use_batch_norm: bool = True):
        super().__init__()
        layers = [
            Flatten(),
            nn.Linear(28 * 28, hidden_size),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 10))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP4(nn.Module):
    """4-layer Multi-Layer Perceptron"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for ResNet-like architecture"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return F.relu(out)


class ResNetLikeCNN(nn.Module):
    """ResNet-like Convolutional Neural Network with residual connections"""
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(16)
        self.pool = nn.MaxPool2d(2, 2)  # 28 -> 14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.block2 = ResidualBlock(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14 -> 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = self.block1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.block2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

