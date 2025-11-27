Loss Landscape Geometry & Mode Connectivity in Wide Neural Networks
==================================================================

This repository provides a modular, reproducible framework for probing the loss landscape of neural networks on MNIST.  
It measures how architectural width and optimization settings influence curvature, flatness, mode connectivity, and generalization.

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
├── main.py            # CLI entry point
├── data.py            # MNIST loaders + subset support
├── models.py          # MLP/CNN architectures (configurable width, BN)
├── training.py        # Config dataclass, schedulers, progress bars
├── landscape.py       # Hessian, sharpness, flatness, connectivity
├── analysis.py        # Experiment orchestration, correlations
├── visualization.py   # Connectivity, correlation, spectrum plots
├── landscape_analysis_results.json  # Latest metrics + histories
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

## Tips for Reproducing Benign Regime

| Hyperparameter | Recommended |
|----------------|-------------|
| Hidden size    | 4096–8192   |
| Optimizer      | AdamW (lr 5e‑4, wd 5e‑4) or SGD (lr 1e‑3, mom 0.85) |
| Batch size     | 128 (adds helpful stochasticity) |
| Scheduler      | Cosine decay to 1e‑5 |
| Epochs         | ≥50 (mode connectivity stabilizes late) |

Smaller widths or overly aggressive LR/batch combinations will revert to sharp, unstable landscapes.

## Future Directions

* Increase seeds (5–10) for statistically meaningful correlations.
* Extend to CNN/ResNet architectures and other datasets.
* Explore non-linear connectivity paths (Bezier, quadratic).
* Compare optimizers under identical training budgets.

## License

MIT License.
