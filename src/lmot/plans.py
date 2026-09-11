"""Transport plans stored as local products, or as a dense matrix.

The LMOT algebra is the same as the validated notebook. Positive residuals at
or below mass_tol are dropped, so marginal errors are always measured.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .common import Fibers, _group_sum, make_fibers, quantile_blocks, check_dense_size

@dataclass
class SlicePlan:
    x: np.ndarray
    y: np.ndarray
    source: Fibers
    target: Fibers
    a: np.ndarray
    b: np.ndarray
    mass: np.ndarray
    common_i: np.ndarray
    common_j: np.ndarray
    common_block: np.ndarray
    h: np.ndarray
    eta: np.ndarray
    mass_tol: float

    def _coeff(self):
        return np.divide(self.mass, self.eta, out=np.zeros_like(self.mass),
                         where=self.eta > self.mass_tol)

    def apply(self, features):
        """Compute Pi @ features without assembling Pi (O((n+m+K)d))."""
        f = np.asarray(features, dtype=float)
        scalar = f.ndim == 1
        if scalar:
            f = f[:, None]
        if f.ndim != 2 or f.shape[0] != len(self.y) or not np.isfinite(f).all():
            raise ValueError("features must have m rows and finite values")
        k = len(self.mass)
        target_mean = _group_sum(self.target.u[:, None] * f,
                                 self.target.groups, self.target.count)
        removed = _group_sum(self.h[:, None] * f[self.common_j],
                             self.common_block, k)
        coeff = self._coeff()[:, None] * (target_mean[self.b] - removed)
        totals = _group_sum(coeff, self.a, self.source.count)
        result = self.source.u[:, None] * totals[self.source.groups]
        corrections = self.h[:, None] * (
            self.mass[self.common_block, None] * f[self.common_j]
            - coeff[self.common_block])
        # Common edges form a partial matching, so source indices are unique.
        result[self.common_i] += corrections
        return result[:, 0] if scalar else result

    def squared_cost(self):
        """Squared Euclidean cost from centered fiber moments.

        Uses the exact residual-product algebra, with a direct residual-moment
        fallback for nearly exhausted blocks to reduce subtractive cancellation.
        No n*m ground-cost matrix is built.
        """
        k = len(self.mass)
        fx, fy = self.source, self.target
        # Translation helps when coordinates are large but their spread is small.
        origin = self.x[0]
        x, y = self.x - origin, self.y - origin
        cx = _group_sum(fx.u[:, None] * x, fx.groups, fx.count)
        cy = _group_sum(fy.u[:, None] * y, fy.groups, fy.count)
        dx, dy = x - cx[fx.groups], y - cy[fy.groups]
        dx2, dy2 = np.einsum("ij,ij->i", dx, dx), np.einsum("ij,ij->i", dy, dy)
        ex = _group_sum(fx.u * dx2, fx.groups, fx.count)
        ey = _group_sum(fy.u * dy2, fy.groups, fy.count)
        sx = _group_sum(fx.u[:, None] * dx, fx.groups, fx.count)[self.a]
        sy = _group_sum(fy.u[:, None] * dy, fy.groups, fy.count)[self.b]
        sx -= _group_sum(self.h[:, None] * dx[self.common_i], self.common_block, k)
        sy -= _group_sum(self.h[:, None] * dy[self.common_j], self.common_block, k)
        ex = ex[self.a] - _group_sum(self.h * dx2[self.common_i], self.common_block, k)
        ey = ey[self.b] - _group_sum(self.h * dy2[self.common_j], self.common_block, k)
        delta = cx[self.a] - cy[self.b]
        val = (ex + ey + self.eta * np.einsum("ij,ij->i", delta, delta)
               + 2 * np.einsum("ij,ij->i", delta, sx - sy))
        val -= np.divide(2 * np.einsum("ij,ij->i", sx, sy), self.eta,
                         out=np.zeros(k), where=self.eta > self.mass_tol)
        val[self.eta <= self.mass_tol] = 0
        fallback = np.flatnonzero((self.eta > self.mass_tol)
                                  & ((self.eta < 1e-8) | (val < 0)))
        for block in fallback:
            ii, jj = fx.indices(self.a[block]), fy.indices(self.b[block])
            r, t = fx.u[ii].copy(), fy.u[jj].copy()
            edge = self.common_block == block
            ri = {int(v): pos for pos, v in enumerate(ii)}
            tj = {int(v): pos for pos, v in enumerate(jj)}
            for i, j, h in zip(self.common_i[edge], self.common_j[edge], self.h[edge]):
                r[ri[int(i)]] -= h
                t[tj[int(j)]] -= h
            rm, tm = r.sum(), t.sum()
            if rm > 0 and tm > 0:
                xr, yt = x[ii], y[jj]
                mr, mt = r @ xr / rm, t @ yt / tm
                vr = r @ np.sum((xr - mr) ** 2, axis=1) / rm
                vt = t @ np.sum((yt - mt) ** 2, axis=1) / tm
                val[block] = rm * tm / self.eta[block] * (vr + vt + np.sum((mr - mt) ** 2))
        diff = x[self.common_i] - y[self.common_j]
        common_cost = np.sum(self.mass[self.common_block] * self.h
                             * np.einsum("ij,ij->i", diff, diff))
        return float(self.mass @ val + common_cost)

    def transpose(self):
        return SlicePlan(self.y, self.x, self.target, self.source, self.b, self.a,
                         self.mass, self.common_j, self.common_i, self.common_block,
                         self.h, self.eta, self.mass_tol)

    def marginals(self):
        return self.apply(np.ones(len(self.y))), self.transpose().apply(np.ones(len(self.x)))

    def entries(self, ii, jj):
        """Read selected coupling entries without materializing the full plan."""
        ii, jj = np.asarray(ii, dtype=int), np.asarray(jj, dtype=int)
        if ii.shape != jj.shape or ii.ndim != 1:
            raise ValueError("entry indices must be equally sized vectors")
        keys = self.a * self.target.count + self.b
        query = self.source.groups[ii] * self.target.count + self.target.groups[jj]
        loc = np.searchsorted(keys, query)
        safe = np.minimum(loc, len(keys)-1)
        valid = (loc < len(keys)) & (keys[safe] == query)
        answer = np.zeros(len(ii))
        i, j, block = ii[valid], jj[valid], loc[valid]
        source_h, target_h = np.zeros(len(self.x)), np.zeros(len(self.y))
        source_block = np.full(len(self.x), -1, dtype=int)
        target_block = np.full(len(self.y), -1, dtype=int)
        match_j = np.full(len(self.x), -1, dtype=int)
        source_h[self.common_i] = target_h[self.common_j] = self.h
        source_block[self.common_i] = target_block[self.common_j] = self.common_block
        match_j[self.common_i] = self.common_j
        r = self.source.u[i] - np.where(source_block[i] == block, source_h[i], 0)
        t = self.target.u[j] - np.where(target_block[j] == block, target_h[j], 0)
        h = np.where(match_j[i] == j, source_h[i], 0)
        answer[valid] = self.mass[block] * h + self._coeff()[block] * r * t
        return answer


@dataclass
class TransportResult:
    squared_cost: float
    barycentric_map: np.ndarray
    plan: object
    diagnostics: dict = field(default_factory=dict)


class SlicedPlan:
    """Uniform average; fibers are streamed and never cached for all slices."""
    def __init__(self, x, alpha, y, beta, projections, common_i, common_j,
                 *, method, fiber_tol=1e-12, mass_tol=1e-12):
        if min(fiber_tol, mass_tol) < 0 or not np.isfinite([fiber_tol, mass_tol]).all():
            raise ValueError("tolerances must be finite and nonnegative")
        self.x, self.alpha, self.y, self.beta = x, alpha, y, beta
        self.projections = projections
        self.common_i, self.common_j = common_i, common_j
        self.method, self.fiber_tol, self.mass_tol = method, fiber_tol, mass_tol

    def iter_slices(self):
        for theta in self.projections:
            source = make_fibers(self.x, self.alpha, theta, self.fiber_tol)
            target = make_fibers(self.y, self.beta, theta, self.fiber_tol)
            a, b, mass = quantile_blocks(source.masses, target.masses)
            i, j = self.common_i, self.common_j
            keys = a * target.count + b
            query = source.groups[i] * target.count + target.groups[j]
            position = np.searchsorted(keys, query)
            safe = np.minimum(position, len(keys)-1)
            valid = (position < len(keys)) & (keys[safe] == query)
            i, j, block = i[valid], j[valid], position[valid]
            h = np.minimum(source.u[i], target.u[j])
            eta = 1 - _group_sum(h, block, len(mass))
            if eta.min() < -1e-11:
                raise RuntimeError("negative residual mass")
            yield SlicePlan(self.x, self.y, source, target, a, b, mass,
                            i, j, block, h, np.maximum(eta, 0), self.mass_tol)

    def evaluate(self):
        origin = self.x[0]
        # Last feature tracks actual row mass, including residual-cutoff effects.
        features = np.column_stack((self.y - origin, np.ones(len(self.y))))
        moment = np.zeros((len(self.x), self.x.shape[1] + 1))
        cost = 0.0
        for part in self.iter_slices():
            moment += part.apply(features)
            cost += part.squared_cost()
        moment /= len(self.projections)
        bary = (moment[:, :-1] + moment[:, -1, None] * origin) / self.alpha[:, None]
        return TransportResult(cost / len(self.projections), bary, self,
                               {"method": self.method, "slices": len(self.projections),
                                "overlap_search": self.method == "LMOT"})

    def apply(self, features):
        total = None
        for part in self.iter_slices():
            value = part.apply(features)
            total = value if total is None else total + value
        return total / len(self.projections)

    def marginals(self):
        rows, cols = np.zeros(len(self.x)), np.zeros(len(self.y))
        for part in self.iter_slices():
            a, b = part.marginals()
            rows += a
            cols += b
        return rows / len(self.projections), cols / len(self.projections)

    def entries(self, i, j):
        return sum(p.entries(i, j) for p in self.iter_slices()) / len(self.projections)

    def to_dense(self, *, max_entries=4_000_000, batch_size=128):
        check_dense_size(len(self.x), len(self.y), max_entries)
        out = np.empty((len(self.x), len(self.y)))
        for start in range(0, len(self.y), batch_size):
            stop = min(start + batch_size, len(self.y))
            basis = np.zeros((len(self.y), stop-start))
            basis[np.arange(start, stop), np.arange(stop-start)] = 1
            out[:, start:stop] = self.apply(basis)
        return out


@dataclass
class DensePlan:
    matrix: np.ndarray

    def apply(self, features):
        return self.matrix @ features

    def marginals(self):
        return self.matrix.sum(axis=1), self.matrix.sum(axis=0)

    def entries(self, i, j):
        return self.matrix[i, j]

    def to_dense(self, *, max_entries=4_000_000, **kwargs):
        check_dense_size(*self.matrix.shape, max_entries)
        return self.matrix.copy()
