# LMOT: Lifted Mask Optimal Transport

Research code for **LMOT with exact binary overlap search**, **EST product
lifting**, and **balanced Sinkhorn**, with a reproducible simulation experiment.
The current scope is finite discrete measures with squared Euclidean ground cost.
MNIST interpolation and color transfer folders are reserved for later experiments.

## Quick start

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m experiments.simulation.run --config experiments/simulation/configs/smoke.yaml
```

The smoke configuration checks the complete pipeline in a small setting. It is
not a paper-scale speed claim. For larger experiments:

```bash
python -m experiments.simulation.run --config experiments/simulation/configs/pilot.yaml
python -m experiments.simulation.run --config experiments/simulation/configs/weights.yaml
python -m experiments.simulation.run --config experiments/simulation/configs/scaling.yaml
```

The pilot and scaling configurations can take substantial time: they sweep
multiple datasets and include small-epsilon Sinkhorn convergence budgets. Edit a
copy of the YAML to select the desired grid. The command prints the run folder.

Recreate figures without rerunning transport:

```bash
python -m experiments.simulation.plot results/simulation/YOUR_RUN_FOLDER
```

For Google Colab, use `notebooks/simulation_colab.ipynb`. Upload and extract the
repository archive, or clone your own GitHub repository, then set `REPO_DIR` in
the notebook. There is no duplicated algorithm implementation in the notebook.

## Repository layout

| Path | Purpose |
|---|---|
| `src/lmot/methods/lmot.py` | Binary overlap search and masked lifting entry point |
| `src/lmot/methods/est.py` | EST entry point; no overlap search |
| `src/lmot/methods/sinkhorn.py` | POT log-domain Sinkhorn and SciPy cross-check backend |
| `src/lmot/common.py` | Input checks, fiber grouping, 1D quantile blocks |
| `src/lmot/plans.py` | Implicit and dense plan interfaces; cost and feature action |
| `src/lmot/projections.py` | Random and structured direction banks |
| `src/lmot/metrics.py` | Overlap, collision, feasibility and exact-OT reference |
| `src/lmot/benchmark.py` | Timing and separate memory measurement |
| `experiments/simulation/` | Data generation, YAMLs, runner, plotting |
| `experiments/mnist/`, `experiments/color_transfer/` | Reserved for experiments 2 and 3 |
| `tests/` | Independent dense reference and correctness checks |
| `notebooks/` | Colab entry point |
| `data/`, `results/` | Data and run artifacts, excluded from git |

## Method API

```python
import numpy as np
from lmot import solve_lmot, solve_est, solve_sinkhorn
from lmot.projections import make_projections

X = np.array([[0., 0.], [0., 1.], [1., 0.]])
Y = np.array([[0., 0.], [0., 1.], [1., 1.]])
alpha = np.array([0.2, 0.3, 0.5])
beta = np.array([0.4, 0.2, 0.4])
theta = make_projections(2, 16, kind="structured", seed=0)

lm = solve_lmot(X, alpha, Y, beta, projections=theta)
est = solve_est(X, alpha, Y, beta, projections=theta)
sk = solve_sinkhorn(X, alpha, Y, beta, epsilon=0.01)

print(lm.squared_cost)
print(lm.barycentric_map)
print(sk.diagnostics)  # Always inspect convergence.
P = lm.plan.to_dense()  # Small problems only; allocation guard is enabled.
```

All solvers return `squared_cost`, `barycentric_map`, `plan`, and `diagnostics`.
All plans support `apply(F)`, `marginals()`, `entries(i,j)`, and `to_dense()`.
Implicit plans rebuild slice groups when queried after solving; these additional
inspection operations are outside the cost/map timer. `to_dense()` is an
inspection/export API, not the intended large-scale interface.

Input arrays are copied to float64. Each measure must have distinct support
atoms, strictly positive weights, and weights summing to one. Merge duplicate
atoms and sum weights before solving; discard zero-weight atoms explicitly.
The core solver does not silently quantize coordinates, add jitter, or modify
weights. EST uses the same validation as LMOT, but never constructs an overlap
index. A new solve call builds a fresh LMOT index for that pair.

## What is implemented mathematically?

For projected fibers I_a and J_b, let u and v be their conditional weights,
and lambda_ab the monotone 1D coupling mass. The implementations use:

- EST: `Q = outer(u, v)`.
- LMOT: `H_ij = 1{x_i == y_j} * min(u_i, v_j)`,
  `r = u - H @ 1`, `t = v - H.T @ 1`, `eta = 1 - H.sum()`,
  `Q = H + outer(r, t) / eta` when residual mass remains.
- The global block is `lambda_ab * Q`.

Binary search stores only exact matching pairs. Common edges are sparse;
residual products stay implicit. Squared cost uses centered moments, and the
map uses the action of the SAME coupling on target coordinates. No self-case
shortcut returns a precomputed identity plan. EST is a reimplementation of the
general discrete product lifting rule, not a claim to reproduce all code or
temperature variants in the EST authors' repository.

Numerical settings preserved from the notebook:

- `fiber_tol=1e-12`: anchored grouping of projected values.
- Exact equality of all ground coordinates for the mask; no near-overlap mask.
- `mass_tol=1e-12`: residual products at or below this threshold are omitted.
  This can lose a small amount of mass; reported marginal errors detect it.
- Very small residual cost uses a direct residual-moment fallback.

The formulas are algebraically equivalent in exact arithmetic. Do not infer a
universal numerical guarantee for arbitrarily tiny weights from the included
tests. This implementation does not establish triangle inequality or a universal
cost improvement over EST.

## Simulation protocol

`scenario: support` starts from a fixed source and replaces a nested subset of
its points with new points from a disjoint pool. Source/target indices are
independently permuted. The weights attached to unchanged points are preserved;
100% requested overlap is therefore a self-case.

`scenario: weights` keeps all locations while setting
`beta = (1-delta)*alpha + delta*b`, where b is a fixed random weight vector.
Here support overlap is always 100%, but common mass changes.

Both measures use ONE geometric scale: coordinates lie in `[0,1/sqrt(d)]^d`.
Squared ground costs are therefore at most one. No per-measure centering or
rescaling is applied. Epsilon is in these squared-cost units. `geometry` can be
`grid` or `continuous`; grid is the default.

Each seed uses exactly the same direction bank for LMOT and EST. Smaller L
values take prefixes of the largest saved bank. Random directions are uniform
on the sphere via normalized Gaussians. Structured directions are canonical
primitive integer directions (including coordinate axes), normalized to unit
length. The structured bank is not a Monte Carlo sample from uniform directions.

A shared source/target atom and a projection collision are different events.
Without exact overlap, LMOT equals EST. When all fibers are singletons, the two
lifts also agree. For finite distinct supports, random continuous directions
have no exact collisions almost surely. Report the measured collision columns;
do not enlarge the grouping tolerance just to obtain an apparent benefit.

### Sinkhorn implementation and convergence

The default `backend: pot` wraps POT's `sinkhorn_log`. Its internal L2
stopping threshold is converted conservatively from the requested L1 marginal
tolerance; both marginals are then checked independently. POT is installed with
the main package dependencies. The iteration cap is its computation budget;
`max_seconds` must be null for this backend.

An independent `backend: scipy_log` is supplied for cross-checking the standard
alternating logsumexp updates. It supports a soft wall-time budget. Tests compare
both backends. Ground cost excludes entropy and is not a debiased Sinkhorn
divergence. These are CPU dense implementations, not a claim to use the fastest
possible GPU or matrix-free Sinkhorn baseline. Benchmark on your final machine
before making a paper-level speed claim.

The SciPy backend checks its time budget at convergence-check intervals; it is
not a hard OS-level timeout. Dense memory is limited by an explicit entry cap.
Rows above the cap are `skipped_size`, NOT a measured out-of-memory event. This
repo does not implement matrix-free Sinkhorn; its size cap does not show that
all Sinkhorn implementations are unable to handle that scale.

### Timing and outputs

Each timer includes input validation/copies, LMOT overlap-index construction,
projection work or Sinkhorn cost construction, and the requested cost and map.
Warmups precede alternating/rotated timed repetitions. Data generation, exact
reference, quality diagnostics, plotting, and file writing are outside timers.
Sinkhorn runs once per dataset/epsilon, not once per projection setting.

Memory is measured in a separate untimed call with `tracemalloc`. It captures
traced allocations (including tracked NumPy buffers), not whole-process peak RSS
or all native-library allocations. Do not label it as GPU memory or total RSS.

Each fresh run directory contains:

| File | Meaning |
|---|---|
| `config.yaml`, `metadata.json` | Exact config, library versions, hardware info, source hash and git state |
| `projections/*.npy` | Shared banks used by both sliced methods |
| `inputs/*.npz` | Exact arrays, if `save_inputs: true` |
| `summary.csv`, `summary.tsv` | One quality/timing row per configuration and seed |
| `raw_timings.csv` | Individual timed repetitions and their status |
| `figures/*.png`, `figures/*.pdf` | Cost, runtime, retention and quality–runtime plots |
| `DONE.json` | Completion marker for the numerical run |

Files are checkpointed after each dataset. Existing output folders are refused;
create a new run folder rather than silently overwriting measurements. Automatic
resume is not implemented.

`cost` is the ground cost; `common_mass` is sum_z min(mu(z),nu(z));
`retained_mass` is the mass actually coupled at identical ground coordinates.
`retention_ratio` is undefined when common mass is zero. Retention is a diagnostic,
not a universal substitute for transport quality.

`relative_gap = (cost - exact_cost)/exact_cost` is reported only for feasible
plans with a positive exact reference cost. Self-cases have no relative gap;
use self-cost and self-map error. Exact OT is solved via a sparse linear program
with SciPy HiGHS at small sizes. It is a numerical reference, with its own solver
tolerance. Exact-OT time is not mixed into method runtime.

Statuses include `ok`, `not_converged`, `timeout`, `invalid_marginals`,
`mixed_timing_status`, and `skipped_size`. All remain in CSV. Figures include only
valid/converged configurations and state how many rows were excluded. Bands are
one standard deviation across data seeds (not confidence intervals). No
statistical significance is inferred from the smoke run.

## Tests and reproducibility

The dense reference in `tests/reference_lmot.py` has independent fiber grouping,
interval intersections, explicit masks, local couplings, and pairwise costs.
Tests compare the full coupling, marginals, projected block masses, selected
entries, feature action, cost, and map. They also cover full collapse, independent
permutations, absent overlap, singleton fibers, small residuals, large offsets,
input validation, analytic Sinkhorn, and nonconvergence reporting.

GitHub Actions runs unit tests (including POT) and the smoke
pipeline on Python 3.11 and 3.12. See `VALIDATION.md` for what was actually run
locally. The repository contains no downloaded datasets, credentials, or
hard-coded GitHub repository URL.

## References

- [Expected Sliced Transport Plans](https://arxiv.org/abs/2410.12176)
- [POT Sinkhorn documentation](https://pythonot.github.io/gen_modules/ot.bregman.html)
- [SciPy linprog documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
