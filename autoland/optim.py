"""Optimizer and scheduler builders for AutoLand."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float, momentum: float = 0.9) -> torch.optim.Optimizer:
    n = name.lower()
    params = model.parameters()
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int, min_lr: float = 1e-5):
    n = name.lower()
    if n == "none":
        return None
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if n == "step":
        step_size = max(epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if n == "onecycle":
        max_lr = max(group["lr"] for group in optimizer.param_groups)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=0.3,
        )
    raise ValueError(f"Unsupported scheduler '{name}'")
