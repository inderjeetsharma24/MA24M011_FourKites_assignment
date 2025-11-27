"""
Utility Functions for Parameter Manipulation
"""

from typing import List
import torch
import torch.nn as nn


def get_params_list(model: nn.Module) -> List[torch.Tensor]:
    """Get list of trainable parameters from model"""
    return [p for p in model.parameters() if p.requires_grad]


def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
    """Flatten list of parameter tensors into single vector"""
    return torch.cat([p.contiguous().view(-1) for p in params])


def assign_flat_params(params: List[torch.Tensor], flat: torch.Tensor):
    """
    In-place assign from flat vector into model parameters.
    
    Args:
        params: List of parameter tensors
        flat: Flat tensor to assign from
    """
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p))
        offset += numel

