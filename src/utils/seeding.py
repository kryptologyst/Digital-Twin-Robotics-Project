"""
Utility functions for deterministic seeding.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


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
