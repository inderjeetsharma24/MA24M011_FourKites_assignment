"""Configuration utilities for the AutoLand loss landscape framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentConfig:
    name: str
    architecture: str
    dataset: str = "mnist"
    hidden_size: int = 1024
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    subset_size: int | None = None
    num_seeds: int = 3
    comprehensive: bool = True
    plot: bool = True
    save_results: bool = True
    compare_optimizers: bool = False
    slice_2d: bool = False
    fast_mode: bool = False


@dataclass
class ExperimentSuite:
    output_dir: Path
    configs: List[ExperimentConfig] = field(default_factory=list)

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["plots", "connectivity", "surfaces", "results"]:
            (self.output_dir / sub).mkdir(exist_ok=True)

    def to_dict(self) -> Dict:
        return {
            "output_dir": str(self.output_dir),
            "configs": [cfg.__dict__ for cfg in self.configs],
        }
