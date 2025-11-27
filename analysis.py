"""
Analysis Framework for Running Experiments
"""

import copy
import json
from typing import Dict, List, Optional, Callable
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from data import get_mnist_loaders
from training import train_model, TrainConfig
from landscape import (
    LossLandscapeAnalyzer, LandscapeMetrics,
    estimate_lambda_max, estimate_trace_hessian, 
    epsilon_sharpness, weight_noise_robustness,
    mode_connectivity_curve
)


def run_experiments_for_model(
    model_name: str, 
    model_fn: Callable, 
    cfg: TrainConfig,
    use_comprehensive_analysis: bool = True,
    subset_size: Optional[int] = None,
    num_seeds: int = 2
) -> Dict:
    """
    Run comprehensive experiments for a model architecture.
    Trains multiple seeds, computes landscape metrics, and analyzes connectivity.
    """
    train_loader, test_loader = get_mnist_loaders(cfg.batch_size, subset_size=subset_size)
    device = cfg.device

    results = []
    state_dicts = []
    all_metrics = []
    test_accs = []
    test_losses = []

    analyzer = LossLandscapeAnalyzer(device=device) if use_comprehensive_analysis else None

    seeds = list(range(num_seeds))
    total_steps = num_seeds
    if HAS_TQDM:
        seed_iter = tqdm(seeds, desc=f"Training {model_name}", unit="seed", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        seed_iter = seeds
    
    for seed_idx, seed in enumerate(seed_iter):
        progress_pct = ((seed_idx + 1) / total_steps) * 100
        if HAS_TQDM:
            seed_iter.set_description(f"Training {model_name} (seed {seed+1}/{num_seeds})")
            seed_iter.set_postfix({'progress': f'{progress_pct:.1f}%'})
        else:
            print(f"\n{'='*60}")
            print(f"Training {model_name} - Seed {seed} ({seed_idx+1}/{num_seeds}) [{progress_pct:.1f}%]")
            print(f"{'='*60}")
        model = model_fn()
        model, history = train_model(model, train_loader, test_loader, cfg, seed=seed)
        state_dicts.append(copy.deepcopy(model.state_dict()))

        test_accs.append(history['test_acc'][-1])
        test_losses.append(history['test_loss'][-1])

        if use_comprehensive_analysis and analyzer:
            fast_mode = getattr(cfg, 'fast_mode', False)
            metrics = analyzer.compute_all_metrics(
                model, train_loader, test_loader, top_k_eigenvalues=10, verbose=True, fast_mode=fast_mode
            )
            all_metrics.append(metrics)

            print(
                f"[{model_name}, seed {seed}] "
                f"lambda_max={metrics.lambda_max:.3e}, lambda_min={metrics.lambda_min:.3e}, "
                f"trace={metrics.trace:.3e}, cond={metrics.condition_number:.2e}, "
                f"sharpness={metrics.epsilon_sharpness:.4f}, "
                f"flatness={metrics.flatness_measure:.4f}, "
                f"noise_drop={metrics.weight_noise_robustness:.4f}, "
                f"grad_norm={metrics.gradient_norm:.3e}, "
                f"eff_rank={metrics.effective_rank:.2f}, "
                f"test_acc={history['test_acc'][-1]:.4f}"
            )

            results.append({
                "history": history,
                "metrics": metrics.to_dict(),
                "final_test_acc": history["test_acc"][-1],
                "final_test_loss": history["test_loss"][-1],
            })
        else:
            # Basic metrics (backward compatibility)
            lambda_max = estimate_lambda_max(model, train_loader, device)
            trace_h = estimate_trace_hessian(model, train_loader, device)
            sharp = epsilon_sharpness(model, train_loader, device)
            robustness = weight_noise_robustness(model, test_loader, device)

            print(
                f"[{model_name}, seed {seed}] "
                f"lambda_max={lambda_max:.3e}, trace={trace_h:.3e}, "
                f"sharpness={sharp:.4f}, noise_drop={robustness:.4f}, "
                f"test_acc={history['test_acc'][-1]:.4f}"
            )

            results.append({
                "history": history,
                "lambda_max": lambda_max,
                "trace": trace_h,
                "sharpness": sharp,
                "noise_drop": robustness,
                "final_test_acc": history["test_acc"][-1],
            })

    # Mode connectivity analysis
    if len(state_dicts) >= 2:
        print(f"\n=== Mode connectivity for {model_name} ===")
        if use_comprehensive_analysis and analyzer:
            connectivity = analyzer.analyze_mode_connectivity(
                model_fn, state_dicts[0], state_dicts[1], test_loader, num_points=21
            )
            print(f"Barrier height: {connectivity['barrier_height']:.4f}")
            print(f"Max loss on path: {connectivity['max_loss_on_path']:.4f}")
            curve = connectivity['connectivity_curve']
        else:
            curve = mode_connectivity_curve(
                model_fn, state_dicts[0], state_dicts[1], test_loader, device, num_points=21
            )
            max_loss_on_path = max(loss for _, loss in curve)
            print(f"Max loss along path: {max_loss_on_path:.4f}")
    else:
        curve = []

    # Correlation analysis if we have multiple seeds with comprehensive metrics
    correlations = {}
    if use_comprehensive_analysis and analyzer and len(all_metrics) >= 2:
        print(f"\n=== Correlation Analysis for {model_name} ===")
        print(f"  Note: Correlations with only {len(all_metrics)} samples are not statistically meaningful.")
        print(f"  Use more seeds (e.g., 5-10) for reliable correlation estimates.")
        correlations = analyzer.correlate_metrics_with_generalization(
            all_metrics, test_accs, test_losses
        )
        for metric_name, corr_value in correlations.items():
            print(f"  {metric_name}: {corr_value:.4f}")

    return {
        'results': results,
        'connectivity_curve': curve,
        'correlations': correlations,
        'model_name': model_name
    }


def compare_optimizers(
    model_name: str,
    model_fn: Callable,
    cfg: TrainConfig,
    optimizers: List[str] = ["SGD", "Adam"],
    seeds: List[int] = [0, 1],
    subset_size: Optional[int] = None
) -> Dict:
    """
    Compare different optimizers on landscape metrics and generalization.
    
    Returns:
        Dictionary with comparison results for each optimizer
    """
    train_loader, test_loader = get_mnist_loaders(cfg.batch_size, subset_size=subset_size)
    device = cfg.device
    
    analyzer = LossLandscapeAnalyzer(device=device)
    comparison_results = {}
    
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"Comparing Optimizer: {opt_name}")
        print(f"{'='*60}")
        
        opt_results = []
        opt_metrics = []
        opt_test_accs = []
        opt_test_losses = []
        
        for seed in seeds:
            print(f"\n--- Training {model_name} with {opt_name}, seed {seed} ---")
            
            # Create config with specific optimizer
            opt_cfg = TrainConfig(
                epochs=cfg.epochs,
                lr=cfg.lr,
                momentum=cfg.momentum,
                batch_size=cfg.batch_size,
                device=cfg.device,
                optimizer=opt_name,
                weight_decay=cfg.weight_decay,
                scheduler=cfg.scheduler,
                lr_min=getattr(cfg, 'lr_min', 1e-5),
            )
            opt_cfg.fast_mode = getattr(cfg, 'fast_mode', False)
            
            # Train model
            model = model_fn()
            model, history = train_model(model, train_loader, test_loader, opt_cfg, seed=seed)
            
            opt_test_accs.append(history['test_acc'][-1])
            opt_test_losses.append(history['test_loss'][-1])
            
            # Compute landscape metrics
            fast_mode = getattr(cfg, 'fast_mode', False)
            metrics = analyzer.compute_all_metrics(
                model, train_loader, test_loader, top_k_eigenvalues=10, verbose=False, fast_mode=fast_mode
            )
            opt_metrics.append(metrics)
            
            opt_results.append({
                "seed": seed,
                "history": history,
                "metrics": metrics.to_dict(),
                "test_acc": history['test_acc'][-1],
                "test_loss": history['test_loss'][-1],
            })
            
            print(
                f"[{opt_name}, seed {seed}] "
                f"lambda_max={metrics.lambda_max:.3e}, "
                f"sharpness={metrics.epsilon_sharpness:.4f}, "
                f"test_acc={history['test_acc'][-1]:.4f}"
            )
        
        # Compute averages
        import numpy as np
        avg_lambda_max = np.mean([m.lambda_max for m in opt_metrics])
        avg_sharpness = np.mean([m.epsilon_sharpness for m in opt_metrics])
        avg_test_acc = np.mean(opt_test_accs)
        avg_test_loss = np.mean(opt_test_losses)
        
        comparison_results[opt_name] = {
            "results": opt_results,
            "avg_lambda_max": float(avg_lambda_max),
            "avg_sharpness": float(avg_sharpness),
            "avg_test_acc": float(avg_test_acc),
            "avg_test_loss": float(avg_test_loss),
            "all_metrics": [m.to_dict() for m in opt_metrics]
        }
    
    return comparison_results


def save_results(experiment_results: Dict, filename: str = "landscape_results.json"):
    """Save experiment results to JSON file"""
    # Convert to JSON-serializable format
    serializable = {}
    for key, value in experiment_results.items():
        if key == 'results':
            serializable[key] = []
            for r in value:
                r_copy = r.copy()
                if 'history' in r_copy:
                    r_copy['history'] = {k: v for k, v in r_copy['history'].items()}
                serializable[key].append(r_copy)
        elif key == 'connectivity_curve':
            serializable[key] = [(float(t), float(loss)) for t, loss in value]
        else:
            serializable[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {filename}")

