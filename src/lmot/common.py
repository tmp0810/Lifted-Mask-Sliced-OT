"""Shared measure validation, fibers and monotone 1D coupling.

No overlap search is performed in this module.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist

def _points(x):
    x = np.array(x, dtype=np.float64, order="C", copy=True)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("points must be finite, nonempty (n,d) arrays")
    return x


def _records(x):
    # Numeric structured records: lexicographic comparison of ALL coordinates.
    # Unlike byte keys, this treats -0.0 and +0.0 as equal.
    dtype = np.dtype([(f"c{k}", np.float64) for k in range(x.shape[1])])
    return np.ascontiguousarray(x).view(dtype).reshape(-1)


def _group_sum(values, groups, count):
    values = np.asarray(values)
    if values.ndim == 1:
        return np.bincount(groups, weights=values, minlength=count)
    return np.column_stack([
        np.bincount(groups, weights=values[:, k], minlength=count)
        for k in range(values.shape[1])
    ])


def _weights(a, n):
    a = np.array(a, dtype=float, copy=True)
    if a.shape != (n,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("weights must be finite, strictly positive and match support size")
    if not np.isclose(a.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("weights must sum to one; normalize explicitly first")
    return a


@dataclass
class Fibers:
    order: np.ndarray
    starts: np.ndarray
    groups: np.ndarray
    masses: np.ndarray
    u: np.ndarray

    @property
    def count(self):
        return len(self.masses)

    def indices(self, a):
        return self.order[self.starts[a]:self.starts[a + 1]]


def make_fibers(x, weights, theta, atol=1e-12):
    """Sort and group projections; match the reference's anchored tolerance."""
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("fiber tolerance must be finite and nonnegative")
    values = x @ theta
    order = np.argsort(values, kind="stable")
    z = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(z) > atol) + 1, len(z)]
    # A chain of close consecutive values need not share the same anchor.
    # Fast vectorized grouping is valid unless a chain exceeds atol.
    if np.any(z[starts[1:] - 1] - z[starts[:-1]] > atol):
        indices = [0]
        anchor = 0
        for j in range(1, len(z)):
            if z[j] - z[anchor] > atol:
                indices.append(j)
                anchor = j
        starts = np.asarray(indices + [len(z)])
    masses = np.add.reduceat(weights[order], starts[:-1])
    groups = np.empty(len(x), dtype=np.int64)
    groups[order] = np.repeat(np.arange(len(masses)), np.diff(starts))
    return Fibers(order, starts, groups, masses, weights / masses[groups])


def quantile_blocks(a, b):
    """Nonzero intersections of cumulative mass intervals; O(k log k).

    Uses vectorized sorted breakpoints and binary search. Tiny positive
    intersections are retained instead of dropped at an absolute mass cutoff.
    """
    ca, cb = np.cumsum(a), np.cumsum(b)
    ca = np.minimum(ca, 1.0)
    cb = np.minimum(cb, 1.0)
    ca[-1] = cb[-1] = 1.0
    edges = np.unique(np.r_[0.0, ca, cb])
    mass = np.diff(edges)
    ends = edges[1:]
    return (np.searchsorted(ca, ends, side="left"),
            np.searchsorted(cb, ends, side="left"), mass)


def prepare_measures(x, alpha, y, beta):
    x, y = _points(x), _points(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and target dimensions differ")
    alpha, beta = _weights(alpha, len(x)), _weights(beta, len(y))
    for support in (x, y):
        if len(np.unique(_records(support))) != len(support):
            raise ValueError("merge repeated atoms and sum their weights first")
    return x, alpha, y, beta


def prepare_projections(projections, dimension):
    theta = np.array(projections, dtype=float, copy=True)
    if theta.ndim != 2 or theta.shape[1] != dimension or len(theta) == 0:
        raise ValueError("projections must have shape (L,d), L > 0")
    norm = np.linalg.norm(theta, axis=1)
    if not np.isfinite(theta).all() or np.any(norm == 0):
        raise ValueError("projection directions must be finite and nonzero")
    return theta / norm[:, None]


def squared_distances(x, y):
    # Direct coordinate differences avoid cancellation in ||x||²+||y||²-2x.y.
    return cdist(x, y, metric="sqeuclidean")


class SizeLimitError(ValueError):
    """A requested dense operation exceeds an explicit allocation budget."""


def check_dense_size(n, m, max_entries):
    if max_entries is not None and n * m > max_entries:
        raise SizeLimitError(f"dense shape ({n},{m}) exceeds {max_entries} entries")
