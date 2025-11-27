"""
Loss Landscape Analysis Functions
Comprehensive tools for analyzing neural network loss landscape geometry.
"""

import math
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from utils import get_params_list, flatten_params, assign_flat_params
from training import eval_model, TrainConfig


@dataclass
class LandscapeMetrics:
    """Container for loss landscape geometric properties"""
    lambda_max: float = 0.0
    lambda_min: float = 0.0
    trace: float = 0.0
    condition_number: float = 0.0
    epsilon_sharpness: float = 0.0
    weight_noise_robustness: float = 0.0
    gradient_norm: float = 0.0
    effective_rank: float = 0.0
    top_k_eigenvalues: List[float] = field(default_factory=list)
    barrier_height: float = 0.0
    flatness_measure: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'lambda_max': self.lambda_max,
            'lambda_min': self.lambda_min,
            'trace': self.trace,
            'condition_number': self.condition_number,
            'epsilon_sharpness': self.epsilon_sharpness,
            'weight_noise_robustness': self.weight_noise_robustness,
            'gradient_norm': self.gradient_norm,
            'effective_rank': self.effective_rank,
            'top_k_eigenvalues': self.top_k_eigenvalues,
            'barrier_height': self.barrier_height,
            'flatness_measure': self.flatness_measure,
        }


def hessian_vector_product(
    model: nn.Module,
    loss_fn,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    v: torch.Tensor,
    retain_graph: bool = False
) -> torch.Tensor:
    """
    Compute H v where H is Hessian of loss wrt parameters.
    Uses efficient Hessian-vector product via autograd.
    """
    params = get_params_list(model)
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grads = flatten_params(list(grads))

    grad_v = (flat_grads * v).sum()
    Hv = torch.autograd.grad(grad_v, params, retain_graph=retain_graph)
    flat_Hv = flatten_params(list(Hv)).detach()
    return flat_Hv


def compute_gradient_norm(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> float:
    """Compute L2 norm of gradient at current parameters"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    total_grad_norm_sq = 0.0
    n_batches = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        flat_grads = flatten_params(list(grads))
        total_grad_norm_sq += flat_grads.norm().item() ** 2
        n_batches += 1

    return math.sqrt(total_grad_norm_sq / n_batches) if n_batches > 0 else 0.0


def estimate_lambda_max(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    iters: int = 20
) -> float:
    """Estimate maximum eigenvalue using power iteration"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(data_loader))
    x, y = x.to(device), y.to(device)

    params = get_params_list(model)
    flat_params = flatten_params(params)
    dim = flat_params.numel()

    v = torch.randn(dim, device=device)
    v = v / v.norm()

    for _ in range(iters):
        Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=True)
        v = Hv / (Hv.norm() + 1e-12)

    Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=False)
    lambda_max = (v * Hv).sum().item()
    return lambda_max


def estimate_lambda_min(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    iters: int = 20
) -> float:
    """Estimate minimum eigenvalue using shifted power iteration"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(data_loader))
    x, y = x.to(device), y.to(device)

    params = get_params_list(model)
    flat_params = flatten_params(params)
    dim = flat_params.numel()

    # Estimate lambda_max first to get shift
    v_max = torch.randn(dim, device=device)
    v_max = v_max / v_max.norm()
    for _ in range(10):
        Hv = hessian_vector_product(model, criterion, x, y, v_max, retain_graph=True)
        v_max = Hv / (Hv.norm() + 1e-12)
    Hv_max = hessian_vector_product(model, criterion, x, y, v_max, retain_graph=True)
    lambda_max_est = (v_max * Hv_max).sum().item()
    
    # Shift to find minimum
    shift = lambda_max_est * 1.1
    v = torch.randn(dim, device=device)
    v = v / v.norm()
    
    for _ in range(iters):
        Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=True)
        shifted_v = shift * v - Hv
        v = shifted_v / (shifted_v.norm() + 1e-12)

    Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=False)
    shifted_rayleigh = (v * (shift * v - Hv)).sum().item()
    lambda_min = shift - shifted_rayleigh
    
    return lambda_min


def lanczos_eigenvalue_spectrum(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    k: int = 20,
    num_batches: int = 1
) -> Tuple[List[float], List[torch.Tensor]]:
    """Estimate top-k eigenvalues using Lanczos algorithm"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    flat_params = flatten_params(params)
    dim = flat_params.numel()
    
    def Hv_fn(v):
        total_Hv = torch.zeros_like(v)
        count = 0
        for i, (x, y) in enumerate(data_loader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=True)
            total_Hv += Hv
            count += 1
        return total_Hv / count if count > 0 else total_Hv
    
    # Lanczos algorithm
    v0 = torch.randn(dim, device=device)
    v0 = v0 / v0.norm()
    V = [v0]
    alpha = []
    beta = [0.0]
    v_prev = torch.zeros_like(v0)
    v_curr = v0

    for i in range(k):
        w = Hv_fn(v_curr)
        if i > 0:
            w = w - beta[i] * v_prev
        alpha_i = (w * v_curr).sum().item()
        alpha.append(alpha_i)
        w = w - alpha_i * v_curr
        beta_i = w.norm().item()
        beta.append(beta_i)
        if beta_i < 1e-10:
            break
        v_prev = v_curr
        v_curr = w / beta_i
        V.append(v_curr)
    
    # Build tridiagonal matrix
    n = len(alpha)
    T = torch.zeros(n, n, device=device)
    for i in range(n):
        T[i, i] = alpha[i]
        if i < n - 1:
            T[i, i+1] = beta[i+1]
            T[i+1, i] = beta[i+1]
    
    # Compute eigenvalues
    eigenvals, eigenvecs = torch.linalg.eigh(T)
    eigenvals = eigenvals.cpu().tolist()
    
    # Convert eigenvectors back to parameter space
    eigenvecs_param = []
    for i in range(min(k, len(eigenvals))):
        vec = torch.zeros_like(v0)
        for j in range(len(V)):
            vec += eigenvecs[j, i].item() * V[j]
        eigenvecs_param.append(vec)
    
    return eigenvals, eigenvecs_param


def estimate_trace_hessian(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_samples: int = 10,
    num_batches: int = 1
) -> float:
    """Estimate trace of Hessian using Hutchinson's method"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    flat_params = flatten_params(params)
    dim = flat_params.numel()

    trace_est = 0.0
    count = 0
    
    for batch_idx, (x, y) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        
        for _ in range(num_samples):
            v = torch.randint(0, 2, (dim,), device=device).float()
            v = 2 * v - 1  # Rademacher: {-1, 1}
            Hv = hessian_vector_product(model, criterion, x, y, v, retain_graph=True)
            trace_est += (v * Hv).sum().item()
            count += 1

    trace_est /= count if count > 0 else 1
    return trace_est


def compute_effective_rank(eigenvalues: List[float]) -> float:
    """Compute effective rank based on eigenvalue spectrum"""
    if not eigenvalues:
        return 0.0
    
    eigenvals = np.array(sorted(eigenvalues, reverse=True))
    eigenvals = np.maximum(eigenvals, 0)
    eigenvals = eigenvals[eigenvals > 1e-10]
    
    if len(eigenvals) == 0:
        return 0.0
    
    if eigenvals[0] > 1e-10:
        effective_rank_ratio = np.sum(eigenvals) / eigenvals[0]
    else:
        effective_rank_ratio = 0.0
    
    return float(effective_rank_ratio)


def epsilon_sharpness(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    epsilon: float = 0.02,
    num_samples: int = 10,
    use_full_dataset: bool = False
) -> float:
    """Compute epsilon-sharpness: max loss increase in epsilon-ball"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    flat_params = flatten_params(params)
    base_norm = flat_params.norm().item()
    radius = epsilon * base_norm

    # Base loss
    base_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for x, y in data_loader:
            if not use_full_dataset and n_total > 0:
                break
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            base_loss += loss.item() * x.size(0)
            n_total += x.size(0)
    base_loss = base_loss / n_total if n_total > 0 else base_loss

    max_increase = 0.0
    for _ in range(num_samples):
        noise = torch.randn_like(flat_params)
        noise = radius * noise / (noise.norm() + 1e-12)
        backup = flat_params.clone()
        perturbed = backup + noise
        assign_flat_params(params, perturbed)

        loss_pert = 0.0
        n_pert = 0
        with torch.no_grad():
            for x, y in data_loader:
                if not use_full_dataset and n_pert > 0:
                    break
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                loss_pert += loss.item() * x.size(0)
                n_pert += x.size(0)
        loss_pert = loss_pert / n_pert if n_pert > 0 else loss_pert

        increase = loss_pert - base_loss
        max_increase = max(max_increase, increase)
        assign_flat_params(params, backup)

    return max_increase


def compute_flatness_measure(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    num_radii: int = 5,
    num_samples_per_radius: int = 10
) -> float:
    """Compute flatness measure as average loss increase across multiple radii"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    flat_params = flatten_params(params)
    base_norm = flat_params.norm().item()

    # Base loss
    base_loss = 0.0
    n_total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            base_loss += loss.item() * x.size(0)
            n_total += x.size(0)
    base_loss = base_loss / n_total if n_total > 0 else base_loss

    radii = np.linspace(0.01, 0.05, num_radii)
    total_increase = 0.0
    total_count = 0

    for radius_frac in radii:
        radius = radius_frac * base_norm
        for _ in range(num_samples_per_radius):
            noise = torch.randn_like(flat_params)
            noise = radius * noise / (noise.norm() + 1e-12)
            backup = flat_params.clone()
            perturbed = backup + noise
            assign_flat_params(params, perturbed)

            loss_pert = 0.0
            n_pert = 0
            with torch.no_grad():
                for x, y in data_loader:
                    x, y = x.to(device), y.to(device)
                    loss = criterion(model(x), y)
                    loss_pert += loss.item() * x.size(0)
                    n_pert += x.size(0)
            loss_pert = loss_pert / n_pert if n_pert > 0 else loss_pert

            increase = loss_pert - base_loss
            total_increase += increase
            total_count += 1
            assign_flat_params(params, backup)

    return total_increase / total_count if total_count > 0 else 0.0


def weight_noise_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    sigma: float = 0.01,
    num_samples: int = 5
) -> float:
    """Returns average drop in test accuracy when weights are perturbed"""
    base_loss, base_acc = eval_model(model, test_loader, device)
    params = get_params_list(model)
    flat_params = flatten_params(params)
    base_norm = flat_params.norm().item()
    noise_scale = sigma * base_norm

    drops = []
    for _ in range(num_samples):
        noise = torch.randn_like(flat_params)
        noise = noise_scale * noise / (noise.norm() + 1e-12)
        backup = flat_params.clone()
        perturbed = backup + noise
        assign_flat_params(params, perturbed)
        _, acc_pert = eval_model(model, test_loader, device)
        drops.append(base_acc - acc_pert)
        assign_flat_params(params, backup)

    avg_drop = sum(drops) / len(drops)
    return avg_drop


def interpolate_state_dict(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    alpha: float
) -> Dict[str, torch.Tensor]:
    """Returns interpolated state_dict: (1 - alpha)*θ1 + alpha*θ2"""
    new_state = {}
    for k in state_dict1.keys():
        new_state[k] = (1 - alpha) * state_dict1[k] + alpha * state_dict2[k]
    return new_state


def mode_connectivity_curve(
    model_fn,
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    data_loader: DataLoader,
    device: str,
    num_points: int = 21
) -> List[Tuple[float, float]]:
    """Returns list of (t, loss) along linear interpolation path"""
    criterion = nn.CrossEntropyLoss()
    ts = torch.linspace(0.0, 1.0, steps=num_points)
    losses = []

    for t in ts:
        alpha = t.item()
        interp_sd = interpolate_state_dict(state_dict1, state_dict2, alpha)
        model = model_fn().to(device)
        model.load_state_dict(interp_sd)
        model.eval()

        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
        losses.append((alpha, total_loss / n))

    return losses


def compute_loss_barrier(
    model_fn,
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor],
    data_loader: DataLoader,
    device: str,
    num_points: int = 21
) -> float:
    """Compute barrier height: max loss along interpolation path minus min loss at endpoints"""
    curve = mode_connectivity_curve(model_fn, state_dict1, state_dict2, 
                                   data_loader, device, num_points)
    losses = [loss for _, loss in curve]
    max_loss = max(losses)
    min_endpoint_loss = min(losses[0], losses[-1])
    barrier = max_loss - min_endpoint_loss
    return barrier


def generate_random_directions(
    model: nn.Module,
    device: str,
    num_directions: int = 2
) -> List[torch.Tensor]:
    """Generate orthonormal random directions in parameter space"""
    params = get_params_list(model)
    flat_params = flatten_params(params)
    dim = flat_params.numel()
    
    directions = []
    for i in range(num_directions):
        v = torch.randn(dim, device=device)
        # Orthogonalize against previous directions
        for prev_dir in directions:
            v = v - (v * prev_dir).sum() * prev_dir
        # Normalize
        v = v / (v.norm() + 1e-10)
        directions.append(v)
    
    return directions


def compute_2d_landscape_slice(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    grid_size: int = 25,
    radius: float = 0.1,
    directions: Optional[List[torch.Tensor]] = None,
    fast_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute loss on a 2D grid along two random directions"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = get_params_list(model)
    flat_params = flatten_params(params)
    base_norm = flat_params.norm().item()
    
    if directions is None:
        directions = generate_random_directions(model, device, num_directions=2)
    U, V = directions[0], directions[1]
    
    # Create grid (reduce size in fast mode)
    actual_grid_size = 15 if fast_mode else grid_size
    alphas = np.linspace(-radius, radius, actual_grid_size)
    betas = np.linspace(-radius, radius, actual_grid_size)
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    loss_grid = np.zeros_like(alpha_grid)
    
    backup = flat_params.clone()
    
    print(f"Computing 2D landscape slice ({actual_grid_size}x{actual_grid_size} grid)...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbation = (alpha * base_norm * U + beta * base_norm * V)
            perturbed = backup + perturbation
            assign_flat_params(params, perturbed)
            
            # Compute loss (use only first batch in fast mode)
            total_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(data_loader):
                    if fast_mode and batch_idx > 0:
                        break
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    total_loss += loss.item() * x.size(0)
                    n += x.size(0)
            
            loss_grid[j, i] = total_loss / n if n > 0 else 0.0
    
    assign_flat_params(params, backup)
    return alpha_grid, beta_grid, loss_grid


class OptimizationTrajectory:
    """Track optimization dynamics during training"""
    def __init__(self):
        self.steps = []
        self.losses = []
        self.grad_norms = []
        self.param_norms = []
        self.test_losses = []
        self.test_accs = []
    
    def add_step(self, step: int, loss: float, grad_norm: float, 
                 param_norm: float, test_loss: float = None, test_acc: float = None):
        self.steps.append(step)
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.param_norms.append(param_norm)
        if test_loss is not None:
            self.test_losses.append(test_loss)
        if test_acc is not None:
            self.test_accs.append(test_acc)
    
    def get_generalization_gap(self) -> List[float]:
        """Compute generalization gap (test_loss - train_loss) over time"""
        if len(self.test_losses) != len(self.losses):
            return []
        return [test - train for test, train in zip(self.test_losses, self.losses)]
    
    def get_gradient_decay_rate(self) -> float:
        """Estimate exponential decay rate of gradient norm"""
        if len(self.grad_norms) < 2:
            return 0.0
        log_grads = [math.log(g + 1e-10) for g in self.grad_norms]
        n = len(log_grads)
        x = np.arange(n)
        coeffs = np.polyfit(x, log_grads, 1)
        return -coeffs[0]


class LossLandscapeAnalyzer:
    """
    Comprehensive framework for analyzing loss landscape geometry.
    Computes multiple geometric properties and their correlations with generalization.
    """
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_cache = {}
    
    def compute_all_metrics(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        top_k_eigenvalues: int = 10,
        verbose: bool = True,
        fast_mode: bool = False
    ) -> LandscapeMetrics:
        """Compute comprehensive set of landscape metrics"""
        if verbose:
            print("Computing landscape metrics...")
        
        metrics = LandscapeMetrics()
        
        if verbose:
            print("  [1/6] Computing gradient norm...")
        metrics.gradient_norm = compute_gradient_norm(model, train_loader, self.device)
        
        if verbose:
            print("  [2/6] Estimating eigenvalue spectrum...")
        try:
            k = 5 if fast_mode else top_k_eigenvalues
            num_batches = 1 if fast_mode else 3
            eigenvals, _ = lanczos_eigenvalue_spectrum(
                model, train_loader, self.device, k=k, num_batches=num_batches
            )
            if eigenvals and len(eigenvals) > 0:
                eigenvals_sorted = sorted(eigenvals, reverse=True)
                metrics.lambda_max = eigenvals_sorted[0]
                metrics.lambda_min = eigenvals_sorted[-1]
                metrics.top_k_eigenvalues = eigenvals_sorted[:min(len(eigenvals_sorted), top_k_eigenvalues)]
                
                if abs(metrics.lambda_min) > 1e-10:
                    metrics.condition_number = abs(metrics.lambda_max / metrics.lambda_min)
                else:
                    metrics.condition_number = float('inf')
                
                metrics.effective_rank = compute_effective_rank(eigenvals_sorted)
            else:
                metrics.lambda_max = 0.0
                metrics.lambda_min = 0.0
                metrics.condition_number = float('inf')
                metrics.effective_rank = 0.0
                metrics.top_k_eigenvalues = []
        except Exception as e:
            if verbose:
                print(f"  Warning: Eigenvalue computation failed: {e}")
            metrics.lambda_max = estimate_lambda_max(model, train_loader, self.device)
            metrics.lambda_min = 0.0
            metrics.condition_number = float('inf')
            metrics.effective_rank = 0.0
            metrics.top_k_eigenvalues = []
        
        if verbose:
            print("  [3/6] Estimating Hessian trace...")
        trace_samples = 3 if fast_mode else 20
        trace_batches = 1 if fast_mode else 3
        metrics.trace = estimate_trace_hessian(
            model, train_loader, self.device, num_samples=trace_samples, num_batches=trace_batches
        )
        
        if verbose:
            print("  [4/6] Computing sharpness measures...")
        sharpness_samples = 5 if fast_mode else 20
        metrics.epsilon_sharpness = epsilon_sharpness(
            model, train_loader, self.device, epsilon=0.02, num_samples=sharpness_samples
        )
        if not fast_mode:
            metrics.flatness_measure = compute_flatness_measure(
                model, train_loader, self.device, num_radii=5, num_samples_per_radius=10
            )
        else:
            metrics.flatness_measure = compute_flatness_measure(
                model, train_loader, self.device, num_radii=2, num_samples_per_radius=3
            )
        
        if verbose:
            print("  [5/6] Computing weight noise robustness...")
        robustness_samples = 3 if fast_mode else 10
        metrics.weight_noise_robustness = weight_noise_robustness(
            model, test_loader, self.device, sigma=0.01, num_samples=robustness_samples
        )
        
        if verbose:
            print("  [6/6] Landscape metrics computed! [DONE]")
        
        return metrics
    
    def analyze_mode_connectivity(
        self,
        model_fn: Callable,
        state_dict1: Dict[str, torch.Tensor],
        state_dict2: Dict[str, torch.Tensor],
        test_loader: DataLoader,
        num_points: int = 21
    ) -> Dict:
        """Analyze connectivity between two minima"""
        curve = mode_connectivity_curve(
            model_fn, state_dict1, state_dict2, test_loader, self.device, num_points
        )
        barrier = compute_loss_barrier(
            model_fn, state_dict1, state_dict2, test_loader, self.device, num_points
        )
        
        return {
            'connectivity_curve': curve,
            'barrier_height': barrier,
            'max_loss_on_path': max(loss for _, loss in curve),
            'min_loss_on_path': min(loss for _, loss in curve),
        }
    
    def correlate_metrics_with_generalization(
        self,
        metrics_list: List[LandscapeMetrics],
        test_accs: List[float],
        test_losses: List[float]
    ) -> Dict:
        """Compute correlations between landscape metrics and generalization"""
        if len(metrics_list) != len(test_accs) or len(metrics_list) != len(test_losses):
            raise ValueError("Mismatch in list lengths")
        
        metric_dicts = [m.to_dict() for m in metrics_list]
        correlations = {}
        
        for metric_name in metric_dicts[0].keys():
            if metric_name == 'top_k_eigenvalues':
                continue
            
            values = [d[metric_name] for d in metric_dicts]
            
            if len(set(values)) > 1:
                corr_acc = np.corrcoef(values, test_accs)[0, 1]
                correlations[f'{metric_name}_vs_test_acc'] = float(corr_acc)
            
            if len(set(values)) > 1:
                corr_loss = np.corrcoef(values, test_losses)[0, 1]
                correlations[f'{metric_name}_vs_test_loss'] = float(corr_loss)
        
        return correlations
    
    def compute_2d_slice(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        grid_size: int = 25,
        radius: float = 0.1,
        fast_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute 2D landscape slice"""
        return compute_2d_landscape_slice(
            model, data_loader, self.device, grid_size, radius, fast_mode=fast_mode
        )

