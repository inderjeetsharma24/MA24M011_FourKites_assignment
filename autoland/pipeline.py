"""Experiment pipeline handling training, metrics, connectivity, and saving."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import ExperimentConfig
from .data import get_mnist_loaders
from .metrics import collect_metrics, two_d_slice
from .plotting import plot_accuracy, plot_connectivity, plot_metrics, plot_surface
from .training import TrainResult, train_model


def _flatten_params(params: List[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().clone().view(-1) for p in params])


def _assign_flat(params: List[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        numel = p.numel()
        p.data = flat[offset : offset + numel].view_as(p).to(p.device)
        offset += numel


def _mode_connectivity(model_a: torch.nn.Module, model_b: torch.nn.Module, loader, device, steps: int = 21) -> List[Tuple[float, float]]:
    state_a = [p.detach().clone() for p in model_a.parameters()]
    state_b = [p.detach().clone() for p in model_b.parameters()]
    alphas = np.linspace(0.0, 1.0, steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    curve = []

    for alpha in alphas:
        for p, pa, pb in zip(model_a.parameters(), state_a, state_b):
            p.data = ((1 - alpha) * pa + alpha * pb).to(device)
        loss = loss_fn(model_a(images), labels).item()
        curve.append((float(alpha), loss))

    for p, pa in zip(model_a.parameters(), state_a):
        p.data = pa.to(device)
    return curve


def _bezier_connectivity(model_a: torch.nn.Module, model_b: torch.nn.Module, loader, device, steps: int = 41) -> List[Tuple[float, float]]:
    params_a = [p.detach().clone() for p in model_a.parameters()]
    params_b = [p.detach().clone() for p in model_b.parameters()]
    params_m = [(pa + pb) / 2 for pa, pb in zip(params_a, params_b)]
    alphas = np.linspace(0.0, 1.0, steps)
    loss_fn = torch.nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    curve = []
    for alpha in alphas:
        for p, pa, pm, pb in zip(model_a.parameters(), params_a, params_m, params_b):
            blended = ((1 - alpha) ** 2) * pa + 2 * (1 - alpha) * alpha * pm + (alpha**2) * pb
            p.data = blended.to(device)
        loss = loss_fn(model_a(images), labels).item()
        curve.append((float(alpha), loss))
    for p, pa in zip(model_a.parameters(), params_a):
        p.data = pa.to(device)
    return curve


def _two_minima_surface(model_a: torch.nn.Module, model_b: torch.nn.Module, loader, device, grid: int = 31, radius: float = 0.25):
    params_a = [p.detach().clone() for p in model_a.parameters()]
    params_b = [p.detach().clone() for p in model_b.parameters()]
    flat_a = _flatten_params(params_a)
    flat_b = _flatten_params(params_b)
    v1 = flat_b - flat_a
    v2 = torch.randn_like(v1)
    v2 -= (v2 @ v1) / (v1.norm() ** 2 + 1e-12) * v1
    v2 = v2 / (v2.norm() + 1e-12)
    v1 = v1 / (v1.norm() + 1e-12)
    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    losses = np.zeros((grid, grid))
    loss_fn = torch.nn.CrossEntropyLoss()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            flat = flat_a + a * v1 + b * v2
            _assign_flat(list(model_a.parameters()), flat)
            losses[j, i] = loss_fn(model_a(images), labels).item()

    _assign_flat(list(model_a.parameters()), flat_a)
    return alphas, betas, losses


def _correlations(metrics: List[Dict[str, float]], histories: List[Dict[str, List[float]]]) -> Dict[str, float]:
    if not metrics:
        return {}

    def corr(a, b):
        if len(set(a)) < 2:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    lambda_vals = [m["lambda_max"] for m in metrics]
    trace_vals = [m["trace"] for m in metrics]
    sharp_vals = [m["epsilon_sharpness"] for m in metrics]
    acc_vals = [h["test_acc"][-1] for h in histories]
    loss_vals = [h["test_loss"][-1] for h in histories]

    return {
        "lambda_vs_acc": corr(lambda_vals, acc_vals),
        "trace_vs_test_loss": corr(trace_vals, loss_vals),
        "sharpness_vs_acc": corr(sharp_vals, acc_vals),
    }


class LandscapePipeline:
    def __init__(self, cfg: ExperimentConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict:
        history_records: List[Dict[str, List[float]]] = []
        metric_records: List[Dict[str, float]] = []
        seed_models: List[torch.nn.Module] = []
        last_test_loader = None

        for seed in range(self.cfg.num_seeds):
            train_loader, test_loader = get_mnist_loaders(self.cfg.batch_size, self.cfg.subset_size, seed)
            last_test_loader = test_loader
            result = train_model(self.cfg, train_loader, test_loader, seed)
            metrics = collect_metrics(result.model, train_loader, test_loader, self.device)
            history_records.append(result.history)
            metric_records.append(metrics)
            seed_models.append(result.model)
            self._save_seed(seed, result, metrics)

        summary = {
            "config": self.cfg.__dict__,
            "histories": history_records,
            "metrics": metric_records,
            "correlations": _correlations(metric_records, history_records),
        }

        if len(seed_models) >= 2 and last_test_loader is not None:
            curve = _mode_connectivity(seed_models[0], seed_models[1], last_test_loader, self.device)
            summary["connectivity_curve_linear"] = curve
            barrier = max(loss for _, loss in curve) - min(curve[0][1], curve[-1][1])
            summary["barrier_height_linear"] = barrier
            plot_connectivity(curve, self.output_dir / f"{self.cfg.name}_connectivity.png")
            bezier = _bezier_connectivity(seed_models[0], seed_models[1], last_test_loader, self.device)
            summary["connectivity_curve_bezier"] = bezier
            barrier_bezier = max(loss for _, loss in bezier) - min(bezier[0][1], bezier[-1][1])
            summary["barrier_height_bezier"] = barrier_bezier
            plot_connectivity(bezier, self.output_dir / f"{self.cfg.name}_connectivity_bezier.png")
            alphas, betas, losses = _two_minima_surface(seed_models[0], seed_models[1], last_test_loader, self.device)
            summary["two_minima_surface"] = {
                "alphas": alphas.tolist(),
                "betas": betas.tolist(),
                "losses": losses.tolist(),
            }
            plot_surface(alphas, betas, losses, self.output_dir / f"{self.cfg.name}_two_minima_surface.png")

        if self.cfg.slice_2d and last_test_loader is not None:
            alphas, betas, losses = two_d_slice(seed_models[-1], last_test_loader, self.device)
            summary["surface"] = {
                "alphas": alphas.tolist(),
                "betas": betas.tolist(),
                "losses": losses.tolist(),
            }
            plot_surface(alphas, betas, losses, self.output_dir / f"{self.cfg.name}_surface.png")

        summary_path = self.output_dir / f"{self.cfg.name}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _save_seed(self, seed: int, result: TrainResult, metrics: Dict[str, float]) -> None:
        plot_accuracy(result.history, self.output_dir / f"{self.cfg.name}_seed{seed}_acc.png")
        plot_metrics(metrics, self.output_dir / f"{self.cfg.name}_seed{seed}_metrics.png")
