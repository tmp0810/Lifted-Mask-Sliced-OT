"""Nested overlap sweeps with known shared atoms and a common cost scale."""
import numpy as np


def make_pair(n, dimension, *, value, seed, weights="nonuniform",
              scenario="support", geometry="grid"):
    if n < 2 or dimension < 2 or not 0 <= value <= 1:
        raise ValueError("n,d must be >=2 and sweep value in [0,1]")
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, dimension]))
    if geometry == "grid":
        side = max(2, int(np.ceil((2*n)**(1/dimension))))
        while side**dimension < 2*n:
            side += 1
        ids = rng.choice(side**dimension, size=2*n, replace=False)
        pool = np.column_stack(np.unravel_index(ids, (side,)*dimension)).astype(float)
        pool /= (side-1) * np.sqrt(dimension)
    elif geometry == "continuous":
        pool = rng.random((2*n, dimension)) / np.sqrt(dimension)
    else:
        raise ValueError("geometry must be grid or continuous")
    # Same fixed bounding-box scale for BOTH measures: squared cost <= 1.
    x, novel = pool[:n], pool[n:]
    if weights == "uniform":
        alpha = np.full(n, 1/n)
    elif weights == "nonuniform":
        alpha = rng.lognormal(0, .65, n)
        alpha /= alpha.sum()
    else:
        raise ValueError("weights must be uniform or nonuniform")
    beta = alpha.copy()
    ordering = rng.permutation(n)
    y = x.copy()
    if scenario == "support":
        shared = int(round(value*n))
        moved = ordering[shared:]
        y[moved] = novel[moved]
        is_self = shared == n
    elif scenario == "weights":
        alternative = rng.lognormal(0, .65, n)
        alternative /= alternative.sum()
        beta = (1-value)*alpha + value*alternative
        is_self = value == 0
    else:
        raise ValueError("scenario must be support or weights")
    p, q = rng.permutation(n), rng.permutation(n)
    return (x[p], alpha[p], y[q], beta[q]), is_self
