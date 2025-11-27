"""
Main Entry Point for Loss Landscape Analysis
"""

import argparse
from training import TrainConfig, train_model
from models import MLP2, MLP4, SimpleCNN, ResNetLikeCNN
from analysis import run_experiments_for_model, compare_optimizers, save_results
from visualization import (
    plot_connectivity_curve, plot_metric_correlations,
    plot_spectrum_histogram, plot_2d_landscape_slice, plot_optimizer_comparison
)
from landscape import LossLandscapeAnalyzer
from data import get_mnist_loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loss Landscape Analysis')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'MLP2', 'MLP4', 'SimpleCNN', 'ResNetLikeCNN'],
                       help='Model to analyze')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--comprehensive', action='store_true', default=True,
                       help='Use comprehensive analysis (default: True)')
    parser.add_argument('--2d-slice', action='store_true',
                       help='Compute and plot 2D landscape slice')
    parser.add_argument('--spectrum-plot', action='store_true',
                       help='Plot eigenvalue spectrum histogram')
    parser.add_argument('--compare-optimizers', action='store_true',
                       help='Compare SGD vs Adam optimizers')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: reduce samples significantly (5-10x speedup)')
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Use subset of MNIST (e.g., 1000 for quick testing)')
    parser.add_argument('--num-seeds', type=int, default=2,
                       help='Number of random seeds to use (default: 2, recommend 5-10 for correlations)')
    parser.add_argument('--hidden-size', type=int, default=4096,
                       help='Hidden layer size for MLP2 (default: 4096 for benign wide regime)')
    parser.add_argument('--no-batch-norm', action='store_true',
                       help='Disable batch normalization (default: enabled for wide networks)')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer to use (default: AdamW for wide networks)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--momentum', type=float, default=0.85,
                       help='Momentum for SGD (ignored for Adam/AdamW)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay / L2 regularization')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                       help='Minimum LR for cosine scheduler')
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        lr_min=args.lr_min,
    )
    # Add fast_mode to config
    cfg.fast_mode = args.fast

    # Create model factory functions with configurable width
    def make_mlp2():
        return MLP2(hidden_size=args.hidden_size, use_batch_norm=not args.no_batch_norm)
    
    models = {
        "MLP2": make_mlp2,
        "MLP4": MLP4,
        "SimpleCNN": SimpleCNN,
        "ResNetLikeCNN": ResNetLikeCNN,
    }

    all_experiments = {}
    
    # Optimizer comparison (runs separately)
    if args.compare_optimizers:
        print(f"\n{'='*60}")
        print("OPTIMIZER COMPARISON")
        print(f"{'='*60}")
        models_to_compare = models if args.model == 'all' else {args.model: models[args.model]}
        for name, fn in models_to_compare.items():
            print(f"\nComparing optimizers for: {name}")
            subset_size = args.subset_size if args.fast else None
            comparison = compare_optimizers(name, fn, cfg, optimizers=["SGD", "Adam"], 
                                           seeds=list(range(args.num_seeds)), subset_size=subset_size)
            
            if args.plot:
                plot_optimizer_comparison(
                    comparison, name,
                    save_path=f"{name}_optimizer_comparison.png" if args.save_results else None
                )
            
            if args.save_results:
                with open(f"{name}_optimizer_comparison.json", 'w') as f:
                    import json
                    json.dump(comparison, f, indent=2, default=str)
            print(f"\nOptimizer Comparison Summary for {name}:")
            for opt_name, results in comparison.items():
                print(f"  {opt_name}: lambda_max={results['avg_lambda_max']:.3e}, "
                      f"sharpness={results['avg_sharpness']:.4f}, "
                      f"test_acc={results['avg_test_acc']:.4f}")
    else:
        models_to_run = models if args.model == 'all' else {args.model: models[args.model]}
        
        # Default subset size in fast mode if not specified
        if args.fast and args.subset_size is None:
            args.subset_size = 1000  # Use 1000 samples by default in fast mode
        
        total_models = len(models_to_run)
        for model_idx, (name, fn) in enumerate(models_to_run.items()):
            model_progress = ((model_idx + 1) / total_models) * 100
            print(f"\n{'='*60}")
            print(f"Analyzing: {name} ({model_idx+1}/{total_models}) [{model_progress:.1f}%]")
            print(f"{'='*60}")
            
            subset_size = args.subset_size
            experiment_results = run_experiments_for_model(
                name, fn, cfg, use_comprehensive_analysis=args.comprehensive, 
                subset_size=subset_size, num_seeds=args.num_seeds
            )
            all_experiments[name] = experiment_results
            
            # Plot connectivity curve
            if args.plot and experiment_results.get('connectivity_curve'):
                plot_connectivity_curve(
                    experiment_results['connectivity_curve'], 
                    name, 
                    save_path=f"{name}_connectivity.png" if args.save_results else None
                )
            
            # Plot correlations
            if args.plot and experiment_results.get('correlations'):
                plot_metric_correlations(
                    experiment_results['correlations'],
                    name,
                    save_path=f"{name}_correlations.png" if args.save_results else None
                )
            
            # 2D landscape slice
            if args.__dict__.get('2d_slice', False):
                print(f"\nComputing 2D landscape slice for {name}...")
                subset_size = args.subset_size if args.fast else None
                train_loader, test_loader = get_mnist_loaders(cfg.batch_size, subset_size=subset_size)
                model = fn()
                model, _ = train_model(model, train_loader, test_loader, cfg, seed=0)
                analyzer = LossLandscapeAnalyzer(device=cfg.device)
                fast_mode = getattr(cfg, 'fast_mode', False)
                alpha_grid, beta_grid, loss_grid = analyzer.compute_2d_slice(
                    model, test_loader, grid_size=25, radius=0.1, fast_mode=fast_mode
                )
                if args.plot:
                    plot_2d_landscape_slice(
                        alpha_grid, beta_grid, loss_grid, name,
                        save_path=f"{name}_2d_landscape.png" if args.save_results else None
                    )
            
            # Spectrum histogram
            if args.__dict__.get('spectrum_plot', False) or args.plot:
                if args.comprehensive and experiment_results.get('results'):
                    for idx, result in enumerate(experiment_results['results']):
                        if 'metrics' in result and 'top_k_eigenvalues' in result['metrics']:
                            eigenvals = result['metrics']['top_k_eigenvalues']
                            if eigenvals:
                                plot_spectrum_histogram(
                                    eigenvals, f"{name}_seed{idx}",
                                    save_path=f"{name}_seed{idx}_spectrum.png" if args.save_results else None
                                )
        
        # Save all results
        if args.save_results:
            save_results(all_experiments, "landscape_analysis_results.json")
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)

