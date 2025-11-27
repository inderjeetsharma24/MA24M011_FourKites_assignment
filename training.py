"""
Training Functions and Configuration
"""

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


@dataclass
class TrainConfig:
    """Configuration for model training"""
    epochs: int = 50  # Wide networks need longer convergence
    lr: float = 5e-4  # Conservative LR for large width
    momentum: float = 0.85
    batch_size: int = 128  # Smaller batch adds helpful noise
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: str = "SGD"  # "SGD", "Adam", or "AdamW"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "none", "cosine", "step"
    lr_min: float = 1e-5  # Minimum LR for cosine scheduler


def train_one_epoch(model, loader, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


def eval_model(model, loader, device):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def train_model(model, train_loader, test_loader, cfg: TrainConfig, seed: int = 0):
    """
    Train model for specified number of epochs.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        cfg: Training configuration
        seed: Random seed
    
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    torch.manual_seed(seed)
    device = cfg.device
    model = model.to(device)
    
    # Support different optimizers
    optimizer_type = getattr(cfg, 'optimizer', 'SGD').upper()
    if optimizer_type == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif optimizer_type == 'ADAMW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:  # Default to SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )

    scheduler = None
    scheduler_type = getattr(cfg, 'scheduler', 'none').lower()
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=getattr(cfg, 'lr_min', 1e-5)
        )
    elif scheduler_type == 'step':
        step_size = max(cfg.epochs // 3, 1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.1
        )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Create progress bar or use simple loop
    epoch_range = range(cfg.epochs)
    if HAS_TQDM:
        epoch_range = tqdm(epoch_range, desc="Training", unit="epoch", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for epoch in epoch_range:
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = eval_model(model, test_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        # Calculate progress percentage
        progress_pct = ((epoch + 1) / cfg.epochs) * 100
        
        current_lr = optimizer.param_groups[0]["lr"]

        if scheduler:
            scheduler.step()

        if HAS_TQDM:
            # Update progress bar description
            epoch_range.set_postfix({
                'train_acc': f'{tr_acc:.2%}',
                'test_acc': f'{te_acc:.2%}',
                'test_loss': f'{te_loss:.4f}',
                'lr': f'{current_lr:.2e}',
            })
        else:
            # Print with percentage
            print(
                f"[{progress_pct:5.1f}%] Epoch {epoch+1}/{cfg.epochs} | "
                f"Train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                f"Test loss {te_loss:.4f}, acc {te_acc:.4f} | "
                f"lr {current_lr:.2e}"
            )

    # Return trained model + history
    return model, history

