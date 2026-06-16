"""
Utility helpers shared across the project.
"""
from __future__ import annotations

import os
import random

import torch
import numpy as np


def seed_everything(seed: int, benchmark: bool = False) -> None:
    """Fix the random seed for full reproducibility across all devices and backends.

    Sets the seed for Python's built-in :mod:`random`, the ``PYTHONHASHSEED``
    environment variable, PyTorch (CPU and all CUDA devices), and enables
    CuDNN deterministic mode so that convolution algorithms are chosen
    deterministically.

    .. note::
        ``torch.backends.cudnn.deterministic = True`` and
        ``torch.backends.cudnn.benchmark = False`` guarantee reproducible
        results but may reduce throughput on fixed-size inputs where CuDNN
        auto-tuning would otherwise select faster kernels.  Pass
        ``benchmark=True`` to trade full reproducibility for speed: CuDNN will
        profile and cache the fastest convolution algorithm for each layer
        shape after the first few batches.

    Args:
        seed: Integer seed value to use everywhere.
        benchmark: When ``True``, enables ``cudnn.benchmark`` and disables
            ``cudnn.deterministic``.  Recommended when all inputs have a fixed
            spatial size (e.g., 256×256 patches).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False