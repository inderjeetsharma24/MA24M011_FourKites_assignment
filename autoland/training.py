"""Training utilities with deterministic seeding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_model
from .optim import build_optimizer, build_scheduler


@dataclass
class TrainResult:
    model: nn.Module
    history: Dict[str, List[float]] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def _evaluate(model, loader, device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            total += labels.size(0)
    return total_loss / total, total_correct / total


def train_model(cfg, train_loader: DataLoader, test_loader: DataLoader, seed: int) -> TrainResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.architecture, hidden_size=cfg.hidden_size).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.weight_decay, cfg.momentum)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.epochs)

    history = {k: [] for k in ["train_loss", "train_acc", "test_loss", "test_acc"]}
    iterator = tqdm(range(cfg.epochs), desc=f"{cfg.name}-seed{seed}", unit="epoch")

    for _ in iterator:
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = _evaluate(model, test_loader, device)
        if scheduler:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        iterator.set_postfix({"train": f"{train_acc:.3f}", "test": f"{test_acc:.3f}"})

    return TrainResult(model=model, history=history)
