"""Shared direction banks; the prefix of a bank is used for smaller budgets."""
import itertools
import math
import numpy as np


def make_projections(dimension, count, *, kind="random", seed=0):
    if dimension < 2 or count < 1:
        raise ValueError("simulation direction banks require d >= 2 and L >= 1")
    rng = np.random.default_rng(seed)
    if kind == "random":
        directions = rng.normal(size=(count, dimension))
    elif kind == "structured":
        # Canonical primitive integer vectors: no parallel +/- duplicates.
        # Exhaustive low-dimensional shells start with coordinate directions.
        vectors = [tuple(row) for row in np.eye(dimension, dtype=int)]
        seen = set(vectors)
        radius = 1
        while len(vectors) < count:
            if dimension <= 3:
                candidates = []
                for v in itertools.product(range(-radius, radius+1), repeat=dimension):
                    if max(map(abs, v)) != radius or math.gcd(*map(abs, v)) != 1:
                        continue
                    if next(z for z in v if z) < 0:
                        continue
                    candidates.append(v)
                candidates.sort(key=lambda v: (sum(z*z for z in v), v))
            else:
                # Deterministic sampled integer directions in high dimensions;
                # do not enumerate exponentially large coordinate grids.
                candidates = []
                for v in rng.integers(-radius, radius+1, size=(max(64, count), dimension)):
                    if not np.any(v):
                        continue
                    v = v // math.gcd(*map(abs, v))
                    if v[np.flatnonzero(v)[0]] < 0:
                        v = -v
                    candidates.append(tuple(v))
            for v in candidates:
                if v not in seen:
                    vectors.append(v)
                    seen.add(v)
            radius += 1
        directions = np.asarray(vectors[:count], dtype=float)
    else:
        raise ValueError("kind must be random or structured")
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
