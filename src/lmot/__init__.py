"""PyTorch transport of finite measures; CUDA is the default device.

Use device='cpu' explicitly only to run the same backend in correctness tests.
The legacy NumPy transport implementation has been removed.
"""
from .gpu import solve_lmot, solve_est, solve_sinkhorn

__all__ = ["solve_lmot", "solve_est", "solve_sinkhorn"]
__version__ = "0.1.0"
