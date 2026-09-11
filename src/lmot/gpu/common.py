"""Device-side validation, lexicographic binary search, fibers and 1D OT.

No NumPy/CPU fallback for transport calculations. Scalars needed for branching
may synchronize CUDA. float64 is deliberate: float32 can change near ties and
cannot retain the original 1e-12 fiber/mass tolerances reliably.
"""
from dataclasses import dataclass
import math
import torch


class SizeLimitError(ValueError):
    """A dense operation exceeds the configured allocation budget."""


def check_dense_size(n, m, max_entries):
    if max_entries is not None and n * m > max_entries:
        raise SizeLimitError(f"dense shape ({n},{m}) exceeds {max_entries} entries")


def resolve_device(device="cuda"):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices: cuda, cuda:N, or explicit cpu for testing")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. In Colab select Runtime > Change runtime type > GPU. "
                               "Use --device cpu only for correctness tests; no automatic fallback.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device does not exist")
    return device


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float64, device=device).detach().contiguous()


def lexsort(points):
    """Numeric lexicographic row ordering using stable device-side sorts."""
    order = torch.arange(len(points), device=points.device)
    for column in range(points.shape[1] - 1, -1, -1):
        order = order[torch.argsort(points[order, column], stable=True)]
    return order


@torch.no_grad()
def find_exact_overlap(x, y):
    """Sort target tuples, then vectorized binary search over all coordinates.

    No dense n*m cost/mask, hashing, quantization or tolerance matching. The
    iterations are fixed from m; all lower/upper bounds and comparisons stay
    on the input device. Distinct, finite supports are required by the solvers.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(y) == 0:
        raise ValueError("expected nonempty compatible point arrays")
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError("overlap inputs must share device and dtype")
    order = lexsort(y)
    target = y[order]
    lo = torch.zeros(len(x), dtype=torch.long, device=x.device)
    hi = torch.full_like(lo, len(y))
    for _ in range(len(y).bit_length()):
        mid = (lo + hi) // 2
        probe = target[mid.clamp_max(len(y) - 1)]
        prefix_equal = torch.ones(len(x), dtype=torch.bool, device=x.device)
        less = torch.zeros_like(prefix_equal)
        for column in range(x.shape[1]):
            less |= prefix_equal & (probe[:, column] < x[:, column])
            prefix_equal &= probe[:, column] == x[:, column]
        active = lo < hi
        lo = torch.where(active & less, mid + 1, lo)
        hi = torch.where(active & ~less, mid, hi)
    safe = lo.clamp_max(len(y) - 1)
    valid = (lo < len(y)) & torch.all(target[safe] == x, dim=1)
    source = torch.nonzero(valid, as_tuple=True)[0]
    return source, order[safe[source]]


def prepare_measures(x, alpha, y, beta, *, device="cuda"):
    device = resolve_device(device)
    x, alpha, y, beta = (tensor(v, device) for v in (x, alpha, y, beta))
    if x.ndim != 2 or y.ndim != 2 or min(x.shape) == 0 or min(y.shape) == 0:
        raise ValueError("points must be nonempty (n,d) arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and target dimensions differ")
    for points, weights in ((x, alpha), (y, beta)):
        if weights.shape != (len(points),):
            raise ValueError("weights must match support size")
        if not bool(torch.isfinite(points).all() & torch.isfinite(weights).all()
                    & (weights > 0).all() & ((weights.sum() - 1).abs() <= 1e-12)):
            raise ValueError("finite points and positive weights summing to one required")
        ordered = points[lexsort(points)]
        if bool(torch.all(ordered[1:] == ordered[:-1], dim=1).any()):
            raise ValueError("merge repeated atoms and sum their weights first")
    return x, alpha, y, beta


def prepare_projections(projections, dimension, device):
    theta = tensor(projections, device)
    if theta.ndim != 2 or theta.shape[1] != dimension or len(theta) == 0:
        raise ValueError("projections must have shape (L,d), L > 0")
    norms = torch.linalg.vector_norm(theta, dim=1)
    if not bool(torch.isfinite(theta).all() & (norms > 0).all()):
        raise ValueError("finite nonzero directions required")
    return theta / norms[:, None]


def group_sum(values, groups, count):
    out = values.new_zeros((count,) + values.shape[1:])
    return out.index_add_(0, groups, values)


@dataclass
class Fibers:
    order: torch.Tensor
    starts: torch.Tensor
    groups: torch.Tensor
    masses: torch.Tensor
    u: torch.Tensor

    @property
    def count(self):
        return len(self.masses)

    def indices(self, block):
        return self.order[int(self.starts[block]):int(self.starts[block + 1])]


def make_fibers(x, weights, theta, atol=1e-12):
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("fiber tolerance must be finite and nonnegative")
    values = x @ theta
    order = torch.argsort(values, stable=True)
    z = values[order]
    starts = torch.cat((order.new_zeros(1), torch.nonzero(z[1:] - z[:-1] > atol,
                       as_tuple=True)[0] + 1, order.new_tensor([len(z)])))
    if bool((z[starts[1:] - 1] - z[starts[:-1]] > atol).any()):
        # Rare chain of near ties. Preserve anchored grouping, not transitive
        # adjacency clustering. Comparisons stay on device; anchors synchronize.
        anchors, current = [0], 0
        while current < len(z):
            # Subtraction matches the CPU rule even near large coordinates.
            stop = torch.searchsorted(z[current:] - z[current], z.new_tensor(atol), right=True)
            current += int(stop)
            anchors.append(current)
        starts = order.new_tensor(anchors)
    counts = starts[1:] - starts[:-1]
    sorted_groups = torch.repeat_interleave(torch.arange(len(counts), device=x.device),
                                            counts, output_size=len(x))
    groups = torch.empty_like(order)
    groups[order] = sorted_groups
    masses = group_sum(weights[order], sorted_groups, len(counts))
    return Fibers(order, starts, groups, masses, weights / masses[groups])


def quantile_blocks(a, b):
    ca, cb = a.cumsum(0).clamp_max(1), b.cumsum(0).clamp_max(1)
    ca[-1] = cb[-1] = 1
    edges = torch.unique(torch.cat((a.new_zeros(1), ca, cb)), sorted=True)
    ends = edges[1:].contiguous()
    return (torch.searchsorted(ca, ends), torch.searchsorted(cb, ends), edges[1:] - edges[:-1])


def squared_distances(x, y, max_entries=4_000_000):
    check_dense_size(len(x), len(y), max_entries)
    # Direct coordinate differences avoid subtractive cancellation and TF32.
    out = x.new_zeros((len(x), len(y)))
    for k in range(x.shape[1]):
        out += (x[:, k, None] - y[None, :, k]).square()
    return out
