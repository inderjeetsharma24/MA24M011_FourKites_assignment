"""Plotting utilities for training histories and landscape metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(history: Dict[str, List[float]], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["test_acc"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_metrics(metrics: Dict[str, float], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    names = list(metrics.keys())
    values = [metrics[k] for k in names]
    bars = plt.bar(names, values)
    plt.xticks(rotation=30, ha="right")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2e}", ha="center", va="bottom")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_connectivity(curve: List[Tuple[float, float]], path: Path) -> None:
    alphas, losses = zip(*curve)
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, losses, marker="o")
    plt.xlabel("α")
    plt.ylabel("Loss")
    plt.title("Mode connectivity")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_surface(alphas: np.ndarray, betas: np.ndarray, losses: np.ndarray, path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt.figure(figsize=(10, 4))
    ax = plt.subplot(1, 2, 1)
    cs = ax.contourf(alphas, betas, losses, levels=25, cmap="viridis")
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_title("Loss contour")

    ax = plt.subplot(1, 2, 2, projection="3d")
    A, B = np.meshgrid(alphas, betas)
    ax.plot_surface(A, B, losses, cmap="viridis")
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("Loss")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()
