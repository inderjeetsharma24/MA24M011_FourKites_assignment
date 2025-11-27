"""Command-line interface for AutoLand experiments."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .pipeline import LandscapePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoLand loss landscape pipeline")
    parser.add_argument("--name", required=True)
    parser.add_argument("--architecture", required=True, choices=["mlp2", "mlp3", "convnet3", "lenet", "resnet18"])
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "cosine", "step", "onecycle"], default="cosine")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output", type=str, default="autoland_runs")
    parser.add_argument("--slice-2d", action="store_true")
    parser.add_argument("--compare-width", action="store_true")
    parser.add_argument("--width-list", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--compare-optimizers", action="store_true")
    parser.add_argument("--compare-schedulers", action="store_true")
    return parser.parse_args()


def _base_cfg(args) -> ExperimentConfig:
    return ExperimentConfig(
        name=args.name,
        architecture=args.architecture,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        subset_size=args.subset_size,
        num_seeds=args.num_seeds,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        slice_2d=args.slice_2d,
    )


def _extract_summary(summary: Dict) -> Tuple[float, float, float]:
    acc = float(np.mean([h["test_acc"][-1] for h in summary["histories"]]))
    lam = float(np.mean([m["lambda_max"] for m in summary["metrics"]]))
    trace = float(np.mean([m["trace"] for m in summary["metrics"]]))
    return acc, lam, trace


def _plot_curve(xs: List[float], ys: List[float], xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def run_width_sweep(args, base_cfg: ExperimentConfig) -> None:
    widths = [int(w.strip()) for w in args.width_list.split(",") if w.strip()]
    records = []
    output_dir = Path(args.output) / f"{args.name}_width_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    for width in widths:
        cfg = replace(base_cfg, name=f"{args.name}_w{width}", hidden_size=width, slice_2d=True)
        pipeline = LandscapePipeline(cfg, output_dir / cfg.name)
        summary = pipeline.run()
        acc, lam, trace = _extract_summary(summary)
        grad = float(np.mean([m["gradient_norm"] for m in summary["metrics"]]))
        sharp = float(np.mean([m["epsilon_sharpness"] for m in summary["metrics"]]))
        records.append(
            {
                "width": width,
                "test_acc": acc,
                "lambda_max": lam,
                "trace": trace,
                "gradient_norm": grad,
                "epsilon_sharpness": sharp,
            }
        )
    json_path = output_dir / "width_sweep.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    xs = [r["width"] for r in records]
    _plot_curve(xs, [r["test_acc"] for r in records], "Width", "Test Accuracy", output_dir / "width_vs_acc.png")
    _plot_curve(xs, [r["lambda_max"] for r in records], "Width", "lambda_max", output_dir / "width_vs_lambda.png")
    _plot_curve(xs, [r["trace"] for r in records], "Width", "Trace(H)", output_dir / "width_vs_trace.png")
    _plot_curve(xs, [r["gradient_norm"] for r in records], "Width", "Gradient Norm", output_dir / "width_vs_grad.png")
    _plot_curve(xs, [r["epsilon_sharpness"] for r in records], "Width", "epsilon-sharpness", output_dir / "width_vs_sharp.png")


def run_optimizer_sweep(args, base_cfg: ExperimentConfig) -> None:
    optimizers = [
        ("sgd_plain", "sgd", 0.0),
        ("sgd_momentum", "sgd", 0.9),
        ("adam", "adam", base_cfg.momentum),
        ("adamw", "adamw", base_cfg.momentum),
    ]
    output_dir = Path(args.output) / f"{args.name}_optim_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, opt_name, momentum in optimizers:
        cfg = replace(base_cfg, name=f"{args.name}_{label}", optimizer=opt_name, momentum=momentum)
        pipeline = LandscapePipeline(cfg, output_dir / cfg.name)
        summary = pipeline.run()
        acc, lam, trace = _extract_summary(summary)
        records.append({"optimizer": label, "test_acc": acc, "lambda_max": lam, "trace": trace})
    with (output_dir / "optimizer_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    labels = [r["optimizer"] for r in records]
    _plot_curve(labels, [r["test_acc"] for r in records], "Optimizer", "Test Accuracy", output_dir / "optim_vs_acc.png")
    _plot_curve(labels, [r["lambda_max"] for r in records], "Optimizer", "lambda_max", output_dir / "optim_vs_lambda.png")
    _plot_curve(labels, [r["trace"] for r in records], "Optimizer", "Trace(H)", output_dir / "optim_vs_trace.png")


def run_scheduler_sweep(args, base_cfg: ExperimentConfig) -> None:
    schedulers = [("const", "none"), ("step", "step"), ("cosine", "cosine"), ("onecycle", "onecycle")]
    output_dir = Path(args.output) / f"{args.name}_sched_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for label, sched_name in schedulers:
        cfg = replace(base_cfg, name=f"{args.name}_{label}", scheduler=sched_name)
        pipeline = LandscapePipeline(cfg, output_dir / cfg.name)
        summary = pipeline.run()
        acc, lam, trace = _extract_summary(summary)
        records.append({"scheduler": label, "test_acc": acc, "lambda_max": lam, "trace": trace})
    with (output_dir / "scheduler_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    labels = [r["scheduler"] for r in records]
    _plot_curve(labels, [r["test_acc"] for r in records], "Scheduler", "Test Accuracy", output_dir / "sched_vs_acc.png")
    _plot_curve(labels, [r["lambda_max"] for r in records], "Scheduler", "lambda_max", output_dir / "sched_vs_lambda.png")
    _plot_curve(labels, [r["trace"] for r in records], "Scheduler", "Trace(H)", output_dir / "sched_vs_trace.png")


def main() -> None:
    args = parse_args()
    base_cfg = _base_cfg(args)
    if args.compare_width:
        run_width_sweep(args, base_cfg)
        return
    if args.compare_optimizers:
        run_optimizer_sweep(args, base_cfg)
        return
    if args.compare_schedulers:
        run_scheduler_sweep(args, base_cfg)
        return
    output_dir = Path(args.output) / args.name
    pipeline = LandscapePipeline(base_cfg, output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
