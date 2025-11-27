Loss Landscape Geometry & Mode Connectivity in Wide Neural Networks
==================================================================

A modular, research-oriented framework to analyze the loss landscape structure of neural networks.
The goal is to understand how width, optimization, and training dynamics shape curvature, sharpness, flatness, and connectivity of minima.

This repository includes Hessian-based metrics, flatness estimators, mode connectivity interpolation, 2D slices, spectral analysis, and correlation studies — all reproducible and configurable.

Why This Matters ?

Training neural networks is not only about reducing loss — it is fundamentally about finding good regions of the loss landscape.

This project helps understand:

Why wider networks generalize better

Why some minima are flat and others are sharp

How two independently trained models can lie in the same basin

How optimization + architecture jointly shape curvature

This toolkit provides a clean, modular way to run such experiments.

## Key Findings (Wide MLP2: width 4096, AdamW, cosine LR)

* **Flat minima**: λ_max drops to **0.20–0.30**, Hessian trace ≈ **2.4–2.6**, gradient norm ≈ **4.4e‑3 – 4.8e‑3**.
* **High accuracy**: Test accuracy **96.2%–96.5%** (subset 6k train / 1.2k test) with low test loss (~0.18).
* **Connected modes**: Mode connectivity barrier height **0.017** with max path loss 0.195 → minima share a flat basin.
* **Correlations** (3 seeds, low statistical power but consistent signs):
  * flatness_measure vs accuracy: **+0.91**
  * lambda_max vs accuracy: **−0.93**
  * trace vs test loss: **−0.97**

Narrow settings (≤512 units, high LR, large batch) produced λ_max ≈ 1–25 and sharp, double-well connectivity curves, showing the necessity of matching optimization to width.

## Repository Structure

```
project/
│
├── main.py                      # Legacy CLI / framework
├── data.py, models.py, ...      # Original MNIST + landscape stack
├── autoland/                    # New modular pipeline package
│   ├── __init__.py
│   ├── config.py                # ExperimentConfig + suites
│   ├── data.py                  # Deterministic MNIST loaders
│   ├── models.py                # MLP/ConvNet/ResNet builders
│   ├── optim.py                 # Adam/AdamW/SGD + schedulers
│   ├── training.py              # TrainResult + seeded loops
│   ├── metrics.py               # Hessian, sharpness, flatness, etc.
│   ├── pipeline.py              # Multi-seed runs, connectivity, 2D surfaces
│   ├── plotting.py              # Accuracy/metric/connectivity/surface plots
│   └── main.py                  # CLI (width/optimizer/scheduler sweeps)
├── auto_landscape_runner.py     # Standalone entry point wrapper
├── autoland_runs/               # Generated summaries/plots (width sweep, etc.)
├── landscape_analysis_results.json
└── README.md
```

## Running Experiments

1. **Install deps (PyTorch, torchvision, numpy, matplotlib, tqdm).**
2. **Baseline wide run (fast subset)**
   ```bash
   python main.py \
     --model MLP2 \
     --hidden-size 4096 \
     --optimizer AdamW \
     --scheduler cosine \
     --epochs 50 \
     --lr 5e-4 \
     --batch-size 128 \
     --subset-size 6000 \
     --num-seeds 3 \
     --comprehensive \
     --plot \
     --save-results
   ```
3. **Full MNIST**: drop `--subset-size`.
4. **Additional features**
   * `--2d-slice` for 2‑D loss surface
   * `--spectrum-plot` for eigenvalue histograms
   * `--compare-optimizers` to run SGD vs Adam/AdamW with consistent configs

All outputs (metrics JSON, connectivity/correlation PNGs) are versioned in the repo root.

## Metrics Implemented

* **Hessian-based**: Lanczos λ_max / spectrum (with fast mode), Hutchinson trace, effective rank.
* **Sharpness/flatness**: ε-sharpness, multi-radius flatness measure, gradient norm, noise robustness.
* **Mode connectivity**: Linear interpolation loss curve, barrier height, optional 2D slicing.


## Future Directions

* Increase seeds (5–10) for statistically meaningful correlations.
* Extend to CNN/ResNet architectures and other datasets.
* Explore non-linear connectivity paths (Bezier, quadratic).
* Compare optimizers under identical training budgets.
* End-to-end sweeps (width, optimizer, scheduler, nonlinear connectivity, dual-minima 2D grids) are already scripted inside `autoland/`; they simply require extended runtime to execute across all configurations. Once compute time is available, running those commands will produce the full set of figures described in the roadmap (accuracy/λ_max/trace vs width, optimizer and schedule studies, Bezier connectivity, multi-minima surfaces), building directly on the code committed here.


