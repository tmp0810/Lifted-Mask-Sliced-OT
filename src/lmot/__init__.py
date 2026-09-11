"""Public API for squared-Euclidean transport of finite measures."""
from .methods import solve_lmot, solve_est, solve_sinkhorn

__all__ = ["solve_lmot", "solve_est", "solve_sinkhorn"]
__version__ = "0.1.0"
