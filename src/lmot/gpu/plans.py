"""The original LMOT algebra on torch tensors, with implicit and dense outputs.

H = mask * min(u,v); r = u-H1; t = v-H'1; Q = H + r t'/eta.
There is no self-identity shortcut and no change to the maximal-common-mass rule.
As in the CPU code, positive residual mass <= mass_tol is dropped; check marginals.
"""
from dataclasses import dataclass, field
import math
import torch

from ..common import check_dense_size
from .common import Fibers, group_sum, make_fibers, quantile_blocks


@dataclass
class TransportResult:
    squared_cost: torch.Tensor
    barycentric_map: torch.Tensor
    plan: object
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SlicePlan:
    x: torch.Tensor
    y: torch.Tensor
    source: Fibers
    target: Fibers
    a: torch.Tensor
    b: torch.Tensor
    mass: torch.Tensor
    common_i: torch.Tensor
    common_j: torch.Tensor
    common_block: torch.Tensor
    h: torch.Tensor
    eta: torch.Tensor
    mass_tol: float
    _entry_maps: object = field(default=None, init=False, repr=False)

    def _coeff(self):
        active = self.eta > self.mass_tol
        denominator = torch.where(active, self.eta, torch.ones_like(self.eta))
        return torch.where(active, self.mass / denominator, torch.zeros_like(self.mass))

    def apply(self, features):
        f = torch.as_tensor(features, dtype=self.x.dtype, device=self.x.device)
        scalar = f.ndim == 1
        if scalar:
            f = f[:, None]
        if f.ndim != 2 or len(f) != len(self.y):
            raise ValueError("features must have m rows")
        target_mean = group_sum(self.target.u[:, None] * f, self.target.groups, self.target.count)
        removed = group_sum(self.h[:, None] * f[self.common_j], self.common_block, len(self.mass))
        coeff = self._coeff()[:, None] * (target_mean[self.b] - removed)
        totals = group_sum(coeff, self.a, self.source.count)
        result = self.source.u[:, None] * totals[self.source.groups]
        correction = self.h[:, None] * (
            self.mass[self.common_block, None] * f[self.common_j] - coeff[self.common_block])
        # Unique common_i: this is a partial matching, not all pairs of a graph.
        result[self.common_i] += correction
        return result[:, 0] if scalar else result

    def transpose(self):
        return SlicePlan(self.y, self.x, self.target, self.source, self.b, self.a,
                         self.mass, self.common_j, self.common_i, self.common_block,
                         self.h, self.eta, self.mass_tol)

    def marginals(self):
        return (self.apply(self.y.new_ones(len(self.y))),
                self.transpose().apply(self.x.new_ones(len(self.x))))

    def squared_cost(self):
        k, fx, fy = len(self.mass), self.source, self.target
        x, y = self.x - self.x[0], self.y - self.x[0]
        cx = group_sum(fx.u[:, None] * x, fx.groups, fx.count)
        cy = group_sum(fy.u[:, None] * y, fy.groups, fy.count)
        dx, dy = x - cx[fx.groups], y - cy[fy.groups]
        dx2, dy2 = dx.square().sum(1), dy.square().sum(1)
        ex = group_sum(fx.u * dx2, fx.groups, fx.count)[self.a]
        ey = group_sum(fy.u * dy2, fy.groups, fy.count)[self.b]
        sx = group_sum(fx.u[:, None] * dx, fx.groups, fx.count)[self.a]
        sy = group_sum(fy.u[:, None] * dy, fy.groups, fy.count)[self.b]
        sx -= group_sum(self.h[:, None] * dx[self.common_i], self.common_block, k)
        sy -= group_sum(self.h[:, None] * dy[self.common_j], self.common_block, k)
        ex -= group_sum(self.h * dx2[self.common_i], self.common_block, k)
        ey -= group_sum(self.h * dy2[self.common_j], self.common_block, k)
        delta = cx[self.a] - cy[self.b]
        active = self.eta > self.mass_tol
        denominator = torch.where(active, self.eta, torch.ones_like(self.eta))
        val = (ex + ey + self.eta * delta.square().sum(1)
               + 2 * (delta * (sx - sy)).sum(1) - 2 * (sx * sy).sum(1) / denominator)
        val = torch.where(active, val, torch.zeros_like(val))
        fallback = torch.nonzero(active & ((self.eta < 1e-8) | (val < 0)), as_tuple=True)[0]
        for block_tensor in fallback:
            # Rare numerical fallback, same residual moment formula as CPU.
            block = int(block_tensor)
            ii, jj = fx.indices(self.a[block]), fy.indices(self.b[block])
            r, t = fx.u[ii].clone(), fy.u[jj].clone()
            edge = self.common_block == block
            source_removed = x.new_zeros(len(x))
            target_removed = y.new_zeros(len(y))
            source_removed[self.common_i[edge]] = self.h[edge]
            target_removed[self.common_j[edge]] = self.h[edge]
            r -= source_removed[ii]
            t -= target_removed[jj]
            rm, tm = r.sum(), t.sum()
            if bool((rm > 0) & (tm > 0)):
                xr, yt = x[ii], y[jj]
                mr, mt = r @ xr / rm, t @ yt / tm
                vr = r @ (xr - mr).square().sum(1) / rm
                vt = t @ (yt - mt).square().sum(1) / tm
                val[block] = rm * tm / self.eta[block] * (vr + vt + (mr - mt).square().sum())
        common_cost = (self.mass[self.common_block] * self.h
                       * (x[self.common_i] - y[self.common_j]).square().sum(1)).sum()
        return self.mass @ val + common_cost

    def entries(self, ii, jj):
        """Vectorized local H + residual product at requested pairs."""
        ii = torch.as_tensor(ii, dtype=torch.long, device=self.x.device)
        jj = torch.as_tensor(jj, dtype=torch.long, device=self.x.device)
        if ii.shape != jj.shape:
            raise ValueError("entry index shapes must agree")
        if self._entry_maps is None:
            sh, th = self.x.new_zeros(len(self.x)), self.y.new_zeros(len(self.y))
            sb = torch.full((len(self.x),), -1, dtype=torch.long, device=self.x.device)
            tb = torch.full((len(self.y),), -1, dtype=torch.long, device=self.x.device)
            match = torch.full_like(sb, -1)
            sh[self.common_i] = th[self.common_j] = self.h
            sb[self.common_i] = tb[self.common_j] = self.common_block
            match[self.common_i] = self.common_j
            self._entry_maps = sh, th, sb, tb, match
        sh, th, sb, tb, match = self._entry_maps
        keys = self.a * self.target.count + self.b
        query = self.source.groups[ii] * self.target.count + self.target.groups[jj]
        loc = torch.searchsorted(keys, query.contiguous())
        safe = loc.clamp_max(len(keys) - 1)
        valid = (loc < len(keys)) & (keys[safe] == query)
        r = self.source.u[ii] - torch.where(sb[ii] == safe, sh[ii], 0.)
        t = self.target.u[jj] - torch.where(tb[jj] == safe, th[jj], 0.)
        h = torch.where(match[ii] == jj, sh[ii], 0.)
        value = self.mass[safe] * h + self._coeff()[safe] * r * t
        return torch.where(valid, value, torch.zeros_like(value))


class SlicedPlan:
    def __init__(self, x, alpha, y, beta, projections, common_i, common_j, *,
                 method, fiber_tol=1e-12, mass_tol=1e-12):
        if min(fiber_tol, mass_tol) < 0 or not all(map(math.isfinite, (fiber_tol, mass_tol))):
            raise ValueError("tolerances must be finite and nonnegative")
        self.x, self.alpha, self.y, self.beta = x, alpha, y, beta
        self.projections, self.common_i, self.common_j = projections, common_i, common_j
        self.method, self.fiber_tol, self.mass_tol = method, fiber_tol, mass_tol

    def iter_slices(self):
        for theta in self.projections:
            source = make_fibers(self.x, self.alpha, theta, self.fiber_tol)
            target = make_fibers(self.y, self.beta, theta, self.fiber_tol)
            a, b, mass = quantile_blocks(source.masses, target.masses)
            i, j = self.common_i, self.common_j
            keys = a * target.count + b
            query = source.groups[i] * target.count + target.groups[j]
            position = torch.searchsorted(keys, query)
            safe = position.clamp_max(len(keys) - 1)
            valid = (position < len(keys)) & (keys[safe] == query)
            i, j, block = i[valid], j[valid], position[valid]
            h = torch.minimum(source.u[i], target.u[j])
            eta = 1 - group_sum(h, block, len(mass))
            if bool((eta < -1e-11).any()):
                raise RuntimeError("negative residual mass")
            yield SlicePlan(self.x, self.y, source, target, a, b, mass, i, j,
                            block, h, eta.clamp_min(0), self.mass_tol)

    def evaluate(self):
        origin = self.x[0]
        features = torch.cat((self.y - origin, self.y.new_ones((len(self.y), 1))), dim=1)
        moment = self.x.new_zeros((len(self.x), self.x.shape[1] + 1))
        cost = self.x.new_zeros(())
        for part in self.iter_slices():
            moment += part.apply(features)
            cost += part.squared_cost()
        moment /= len(self.projections)
        bary = (moment[:, :-1] + moment[:, -1, None] * origin) / self.alpha[:, None]
        return TransportResult(cost / len(self.projections), bary, self,
                               {"method": self.method, "backend": "torch", "status": "ok",
                                "device": str(self.x.device), "slices": len(self.projections),
                                "overlap_search": self.method == "LMOT"})

    def apply(self, features):
        return sum(part.apply(features) for part in self.iter_slices()) / len(self.projections)

    def marginals(self):
        rows, cols = self.x.new_zeros(len(self.x)), self.y.new_zeros(len(self.y))
        for part in self.iter_slices():
            a, b = part.marginals()
            rows += a
            cols += b
        return rows / len(self.projections), cols / len(self.projections)

    def entries(self, i, j):
        return sum(part.entries(i, j) for part in self.iter_slices()) / len(self.projections)

    def to_dense(self, *, max_entries=4_000_000, batch_size=128):
        check_dense_size(len(self.x), len(self.y), max_entries)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        out = self.x.new_zeros((len(self.x), len(self.y)))
        columns = torch.arange(len(self.y), device=self.x.device)
        # Stream one slice and row tile at a time. No L*n*m stack, no repeated
        # projection/sort for every tile, and no application to a dense identity.
        for part in self.iter_slices():
            for start in range(0, len(self.x), batch_size):
                stop = min(start + batch_size, len(self.x))
                rows = torch.arange(start, stop, device=self.x.device)
                ii, jj = torch.meshgrid(rows, columns, indexing="ij")
                out[start:stop] += part.entries(ii, jj)
        return out / len(self.projections)


@dataclass
class DensePlan:
    matrix: torch.Tensor

    def apply(self, features):
        return self.matrix @ torch.as_tensor(features, device=self.matrix.device, dtype=self.matrix.dtype)

    def marginals(self):
        return self.matrix.sum(1), self.matrix.sum(0)

    def entries(self, i, j):
        return self.matrix[i, j]

    def to_dense(self, *, max_entries=4_000_000, **kwargs):
        check_dense_size(*self.matrix.shape, max_entries)
        return self.matrix.clone()
