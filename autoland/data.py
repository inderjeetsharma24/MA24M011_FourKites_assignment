"""Deterministic data loading utilities."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)


def get_mnist_loaders(
    batch_size: int,
    subset_size: Optional[int],
    seed: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if subset_size and subset_size < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=generator)[:subset_size]
        train_ds = Subset(train_ds, indices.tolist())
        test_subset = min(max(subset_size // 6, 1000), len(test_ds))
        test_idx = torch.randperm(len(test_ds), generator=generator)[:test_subset]
        test_ds = Subset(test_ds, test_idx.tolist())

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, test_loader
