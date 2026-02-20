"""
Utility functions for device management and deterministic seeding.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available computing device.
    
    Priority: CUDA -> MPS (Apple Silicon) -> CPU
    
    Args:
        device: Preferred device ('cuda', 'mps', 'cpu')
        
    Returns:
        torch.device: The selected device
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_deterministic_seeds(seed: int) -> None:
    """
    Set deterministic seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Enable deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)
