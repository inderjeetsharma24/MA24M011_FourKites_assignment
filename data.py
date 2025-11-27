"""
Data Loading Utilities
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size: int = 128, subset_size: Optional[int] = None):
    """
    Get MNIST data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        subset_size: If provided, use only that many samples for faster training
    
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # standard MNIST stats
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    
    # Use subset if specified
    if subset_size is not None:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        test_subset_size = min(subset_size // 5, len(test_dataset))  # Smaller test set
        test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        print(f"Using subset: {len(train_dataset)} train, {len(test_dataset)} test samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

