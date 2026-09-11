"""PyTorch float64 backend. CUDA by default; explicit CPU mode for parity tests."""
from .methods import solve_lmot, solve_est, solve_sinkhorn
from .common import find_exact_overlap, resolve_device, synchronize

__all__ = ["solve_lmot", "solve_est", "solve_sinkhorn", "find_exact_overlap",
           "resolve_device", "synchronize"]
