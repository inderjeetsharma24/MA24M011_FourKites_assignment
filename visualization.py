"""
Visualization Functions for Loss Landscape Analysis
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")


def plot_connectivity_curve(curve: List[Tuple[float, float]], model_name: str, save_path: str = None):
    """Plot mode connectivity curve"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping plot.")
        return
    
    ts, losses = zip(*curve)
    plt.figure(figsize=(8, 6))
    plt.plot(ts, losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Interpolation Parameter α', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title(f'Mode Connectivity: {model_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_metric_correlations(correlations: Dict, model_name: str, save_path: str = None):
    """Plot correlation heatmap"""
    if not HAS_MATPLOTLIB or not correlations:
        return
    
    # Extract metric names and correlation values
    metric_names = []
    corr_values = []
    for key, value in correlations.items():
        if 'vs_test_acc' in key:
            metric_names.append(key.replace('_vs_test_acc', ''))
            corr_values.append(value)
    
    if not metric_names:
        return
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(metric_names))
    colors = ['red' if v < 0 else 'green' for v in corr_values]
    plt.barh(y_pos, corr_values, color=colors, alpha=0.7)
    plt.yticks(y_pos, metric_names)
    plt.xlabel('Correlation with Test Accuracy', fontsize=12)
    plt.title(f'Landscape Metric Correlations: {model_name}', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Correlation plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_spectrum_histogram(
    eigenvalues: List[float],
    model_name: str,
    save_path: Optional[str] = None,
    bins: int = 30,
    log_scale: bool = True
):
    """Plot histogram of eigenvalue spectrum from Lanczos"""
    if not HAS_MATPLOTLIB or not eigenvalues:
        print("Matplotlib not available or no eigenvalues. Skipping plot.")
        return
    
    eigenvals = np.array(eigenvalues)
    eigenvals = eigenvals[eigenvals > 0]  # Filter positive eigenvalues
    
    if len(eigenvals) == 0:
        print("No positive eigenvalues to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale histogram
    ax1.hist(eigenvals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Eigenvalue', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Eigenvalue Spectrum (Linear): {model_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=eigenvals.max(), color='r', linestyle='--', linewidth=2,
                label=f'lambda_max={eigenvals.max():.3e}')
    ax1.axvline(x=eigenvals.min(), color='g', linestyle='--', linewidth=2,
                label=f'lambda_min={eigenvals.min():.3e}')
    ax1.legend()
    
    # Log scale histogram
    if log_scale:
        ax2.hist(eigenvals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.set_xscale('log')
        ax2.set_xlabel('Eigenvalue (log scale)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Eigenvalue Spectrum (Log): {model_name}', fontsize=14)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.axvline(x=eigenvals.max(), color='r', linestyle='--', linewidth=2,
                    label=f'lambda_max={eigenvals.max():.3e}')
        ax2.axvline(x=eigenvals.min(), color='g', linestyle='--', linewidth=2,
                    label=f'lambda_min={eigenvals.min():.3e}')
        ax2.legend()
    else:
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectrum histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_2d_landscape_slice(
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    loss_grid: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    contour_levels: int = 20
):
    """Plot 2D landscape slice as contour and 3D surface"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping plot.")
        return
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("3D plotting not available. Using 2D contour only.")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
    else:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')
    
    # Contour plot
    if 'ax1' in locals():
        contour = ax1.contour(alpha_grid, beta_grid, loss_grid, levels=contour_levels, cmap='viridis')
        ax1.contourf(alpha_grid, beta_grid, loss_grid, levels=contour_levels, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, ax=ax1, label='Loss')
        ax1.set_xlabel('Direction U (α)', fontsize=12)
        ax1.set_ylabel('Direction V (β)', fontsize=12)
        ax1.set_title(f'2D Loss Landscape Contour: {model_name}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax1.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        
        # 3D surface plot
        surf = ax2.plot_surface(alpha_grid, beta_grid, loss_grid, cmap='viridis', 
                               alpha=0.9, linewidth=0, antialiased=True)
        ax2.set_xlabel('Direction U (α)', fontsize=12)
        ax2.set_ylabel('Direction V (β)', fontsize=12)
        ax2.set_zlabel('Loss', fontsize=12)
        ax2.set_title(f'3D Loss Landscape Surface: {model_name}', fontsize=14)
        fig.colorbar(surf, ax=ax2, shrink=0.5, label='Loss')
    else:
        contour = ax.contour(alpha_grid, beta_grid, loss_grid, levels=contour_levels, cmap='viridis')
        ax.contourf(alpha_grid, beta_grid, loss_grid, levels=contour_levels, cmap='viridis', alpha=0.8)
        plt.colorbar(contour, ax=ax, label='Loss')
        ax.set_xlabel('Direction U (α)', fontsize=12)
        ax.set_ylabel('Direction V (β)', fontsize=12)
        ax.set_title(f'2D Loss Landscape: {model_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"2D landscape plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_optimizer_comparison(
    comparison_results: Dict,
    model_name: str,
    save_path: Optional[str] = None
):
    """Plot comparison of optimizers on key metrics"""
    if not HAS_MATPLOTLIB or not comparison_results:
        print("Matplotlib not available or no comparison data. Skipping plot.")
        return
    
    optimizers = list(comparison_results.keys())
    metrics_to_plot = ['lambda_max', 'epsilon_sharpness', 'test_acc']
    metric_labels = ['lambda_max (Hessian)', 'epsilon-Sharpness', 'Test Accuracy']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        values = []
        opt_names = []
        
        for opt_name in optimizers:
            if metric == 'test_acc':
                values.append(comparison_results[opt_name]['avg_test_acc'])
            elif metric == 'lambda_max':
                values.append(comparison_results[opt_name]['avg_lambda_max'])
            elif metric == 'epsilon_sharpness':
                values.append(comparison_results[opt_name]['avg_sharpness'])
            opt_names.append(opt_name)
        
        bars = ax.bar(opt_names, values, alpha=0.7, edgecolor='black')
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label}: {model_name}', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == 'test_acc':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Optimizer comparison plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

