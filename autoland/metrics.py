"""Comprehensive loss landscape metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.contiguous().view(-1) for t in tensors])


def _hvp(loss: torch.Tensor, params: List[torch.nn.Parameter], vector: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = _flatten(grads)
    grad_v = (flat_grads * vector).sum()
    hv = torch.autograd.grad(grad_v, params, retain_graph=True)
    return _flatten(hv)


def top_eigenvalue(model: nn.Module, loader: DataLoader, device: torch.device, iters: int = 20) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    vector = torch.randn_like(_flatten([p.detach() for p in params]))
    vector /= vector.norm() + 1e-12
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    for _ in range(iters):
        logits = model(images)
        loss = loss_fn(logits, labels)
        hv = _hvp(loss, params, vector)
        vector = hv / (hv.norm() + 1e-12)

    logits = model(images)
    loss = loss_fn(logits, labels)
    hv = _hvp(loss, params, vector)
    return (vector * hv).sum().item()


def hessian_trace(model: nn.Module, loader: DataLoader, device: torch.device, samples: int = 20) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    dim = _flatten([p.detach() for p in params]).numel()
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    trace = 0.0

    for _ in range(samples):
        vector = torch.randint(0, 2, (dim,), device=device, dtype=torch.float32) * 2 - 1
        logits = model(images)
        loss = loss_fn(logits, labels)
        hv = _hvp(loss, params, vector)
        trace += (vector * hv).sum().item()

    return trace / samples


def effective_rank(eigenvalues: List[float]) -> float:
    vals = np.array([max(ev, 0.0) for ev in eigenvalues], dtype=float)
    if vals.sum() == 0:
        return 0.0
    probs = vals / vals.sum()
    entropy = -(probs * np.log(probs + 1e-12)).sum()
    return float(np.exp(entropy))


def gradient_norm(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss = loss_fn(logits, labels)
    grads = torch.autograd.grad(loss, params)
    return _flatten(grads).norm().item()


def epsilon_sharpness(model: nn.Module, loader: DataLoader, device: torch.device, epsilon: float = 0.02, samples: int = 5) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    base = _flatten([p.detach().clone() for p in params])
    base_norm = base.norm().item()
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    base_loss = loss_fn(model(images), labels).item()
    radius = epsilon * base_norm
    worst = 0.0

    for _ in range(samples):
        noise = torch.randn_like(base)
        noise = radius * noise / (noise.norm() + 1e-12)
        offset = 0
        for p in params:
            numel = p.numel()
            p.data = (base[offset:offset+numel] + noise[offset:offset+numel]).view_as(p).to(device)
            offset += numel
        loss = loss_fn(model(images), labels).item()
        worst = max(worst, loss - base_loss)

    offset = 0
    for p in params:
        numel = p.numel()
        p.data = base[offset:offset+numel].view_as(p).to(device)
        offset += numel

    return worst


def weight_noise_robustness(model: nn.Module, loader: DataLoader, device: torch.device, sigma: float = 0.01, samples: int = 5) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    base = _flatten([p.detach().clone() for p in params])
    base_norm = base.norm().item()
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    base_loss = loss_fn(model(images), labels).item()
    deltas = []

    for _ in range(samples):
        noise = torch.randn_like(base)
        noise = sigma * base_norm * noise / (noise.norm() + 1e-12)
        offset = 0
        for p in params:
            numel = p.numel()
            p.data = (base[offset:offset+numel] + noise[offset:offset+numel]).view_as(p).to(device)
            offset += numel
        loss = loss_fn(model(images), labels).item()
        deltas.append(loss - base_loss)

    offset = 0
    for p in params:
        numel = p.numel()
        p.data = base[offset:offset+numel].view_as(p).to(device)
        offset += numel

    return float(sum(deltas) / len(deltas))


def two_d_slice(model: nn.Module, loader: DataLoader, device: torch.device, grid: int = 21, radius: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = [p for p in model.parameters() if p.requires_grad]
    base = _flatten([p.detach().clone() for p in params])
    base_norm = base.norm().item()
    directions = [torch.randn_like(base) for _ in range(2)]
    directions = [d / (d.norm() + 1e-12) for d in directions]
    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    losses = np.zeros((grid, grid))
    loss_fn = nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            delta = base_norm * (a * directions[0] + b * directions[1])
            offset = 0
            for p in params:
                numel = p.numel()
                p.data = (base[offset:offset+numel] + delta[offset:offset+numel]).view_as(p).to(device)
                offset += numel
            losses[j, i] = loss_fn(model(images), labels).item()

    offset = 0
    for p in params:
        numel = p.numel()
        p.data = base[offset:offset+numel].view_as(p).to(device)
        offset += numel

    return alphas, betas, losses


def collect_metrics(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    lambda_max = top_eigenvalue(model, train_loader, device)
    trace = hessian_trace(model, train_loader, device)
    grad = gradient_norm(model, train_loader, device)
    sharp = epsilon_sharpness(model, train_loader, device)
    robust = weight_noise_robustness(model, train_loader, device)
    eff_rank = effective_rank([lambda_max, trace / max(lambda_max, 1e-8)])
    return {
        "lambda_max": lambda_max,
        "trace": trace,
        "gradient_norm": grad,
        "epsilon_sharpness": sharp,
        "weight_noise_robustness": robust,
        "effective_rank": eff_rank,
    }
