# CUDA backend: LMOT, EST and Sinkhorn

The transport implementations live in `src/lmot/gpu/`. Both `from lmot import
...` and `from lmot.gpu import ...` expose these same PyTorch solvers with CUDA
as the default device. The legacy NumPy solvers and CPU experiment runners
have been removed. Explicit `device="cpu"` runs the same torch implementation
for correctness tests; there is no automatic fallback.

Shared direction generation remains in `src/lmot/projections.py`. Dataset
creation and configuration/CSV utilities live under `experiments/simulation/`.
The small NumPy oracle under `tests/` is an independent mathematical reference,
not a selectable production backend.

## Colab: installation and correctness check

Select a **GPU runtime** in Colab. Copy the update files into your repository,
then run this cell (adjust the path if necessary):

```python
%cd /content/Lifted-Mask-Sliced-OT
%pip install -e ".[gpu]"

import torch
assert torch.cuda.is_available(), "Select a GPU runtime in Colab first."
print(torch.__version__, torch.cuda.get_device_name(0))

!LMOT_REQUIRE_CUDA=1 python -m unittest tests.test_gpu -v
```

`LMOT_REQUIRE_CUDA=1` causes an error if CUDA is unavailable. A successful
CPU-only test run must not be taken as a completed CUDA validation. The default
experiment device is `cuda`; it never silently falls back to CPU.

PyTorch is now a required dependency, so `pip install -e .` is sufficient.
The existing `.[gpu]` extra and `requirements-gpu.txt` remain installation
aliases. An installed compatible CUDA build in Colab is reused; no CPU wheel
or accelerator-specific download URL is forced.

## Main output: full GPU plan, runtime, RMSE and identity

```python
!python -m experiments.simulation.paper_results_gpu --config experiments/simulation/configs/smoke.yaml --device cuda --reference-epsilon 0.001
```

After the correctness check and smoke run, the existing larger config also works:

```python
!python -m experiments.simulation.paper_results_gpu --config experiments/simulation/configs/pilot.yaml --device cuda --reference-epsilon 0.001
```

Use the same projection bank for LMOT and EST, including identical prefixes for
different values of L. Dataset generation, seeds, exact overlap, structured vs
random directions and normalized squared-Euclidean costs are unchanged.

Outputs go to `results/simulation/paper_gpu_<config-name>_<timestamp>/`:

- `paper_results.csv` and `.tsv`: runtime and plan RMSE, with valid/total counts.
- `identity_results.csv` and `.tsv`: self-plan comparison against the
  coordinate-aligned identity coupling, independent of Sinkhorn regularization.
- `per_pair.csv`: cost, marginals, overlap, collision statistics, statuses and
  median runtime for each pair/method.
- `raw_timings.csv`: each measured repetition, including unsuccessful calls.
- `metadata.json`, `config.yaml`, projection banks and optional input arrays.

All predictions, Sinkhorn reference matrices and RMSE calculations stay on the
selected device. Only scalar diagnostics and final summary values are copied
to the CPU for serialization. Data generation and CSV writing remain host tasks.

### What is timed

The main GPU timing is wall time for **device-resident inputs to a complete
device-resident dense plan**, including the solvers' cost and barycentric-map
outputs. Every timed call performs:

```python
torch.cuda.synchronize(device)
t0 = time.perf_counter()
result = solver(...)          # includes overlap search for LMOT
P = result.plan.to_dense(...) # if the solver's plan is implicit
torch.cuda.synchronize(device)
runtime_ms = 1000 * (time.perf_counter() - t0)
```

Sinkhorn already returns a dense plan, so its matrix is used directly without a
redundant copy. Its ground-cost matrix is constructed inside its timed solve.
LMOT and EST export dense plans by streaming slices and row tiles; no L*n*m
stack is allocated. The GPU export uses direct entry evaluation.

Excluded from all methods' timing: input generation, host-to-device transfer,
projection-bank generation/transfer, the independent reference solve, RMSE,
additional quality checks, device-to-host export and file I/O. Do not compare
this GPU-resident timing against a CPU/GPU round-trip protocol without labeling
the difference. CUDA synchronization includes host orchestration and waits for
queued GPU work; it is not just kernel-enqueue time.

Each pair's runtime is the median of its repetitions; means and sample standard
deviations (ddof=1) are then computed across seeds. SD is blank for one seed.
`measure_memory: true` records extra peak torch CUDA allocation relative to the
allocation at timer entry, including temporary solver/output tensors. This is
not total GPU reserved memory, total device utilization or process RSS.

### Reference and failure handling

Plan error is exactly `sqrt(mean((P - P_gt)**2))` on the probability coupling,
with no row normalization. `P_gt` is an independently converged Sinkhorn plan
at `--reference-epsilon` (default 0.001), checked against both marginals. It is an
entropic reference, not an exact unregularized OT plan or an identity coupling.
The reference default budget is 50,000 iterations with marginal L1 tolerance
1e-9. Baseline budgets and epsilons still come from the YAML config. GPU support
does not remove the need to check convergence or select iteration budgets.

An unconverged reference is never used to score RMSE. Nonconverged predictions
are retained in detailed files and counted as failures in the main aggregate.
Empty aggregate metric/runtime cells are not zeros; raw elapsed time remains
available. Identity metrics for valid lifted plans can still be reported when
the separate Sinkhorn reference failed.

The dense-entry limit is the smaller of `--max-entries` and the config's
`sinkhorn.max_entries`. Default CLI cap: 1,048,576 entries. Thus the existing
pilot's n=4096 dense cases are skipped unless **both** relevant caps permit
them. Raising a cap is a memory-budget decision, not a change to the method.

## Scaling: implicit GPU plan

```python
!python -m experiments.simulation.run_gpu --config experiments/simulation/configs/scaling.yaml --device cuda
```

This measures cost + barycentric map + implicit plan for LMOT/EST. Sinkhorn
remains dense and is explicitly skipped above its allocation limit. Dense RMSE
and a dense identity matrix are not constructed in this mode. Output includes
`summary.csv/.tsv`, detailed cost/marginal/retention/self-map checks and timings.
The columns `device`, `dtype` and `output_mode` distinguish these results from
the dense-plan paper tables. Do not pool the two output protocols.

## Calling the methods directly

```python
import torch
from lmot.gpu import solve_lmot, solve_est, solve_sinkhorn

device = "cuda"
X = torch.as_tensor(X_numpy, dtype=torch.float64, device=device)
a = torch.as_tensor(alpha_numpy, dtype=torch.float64, device=device)
Y = torch.as_tensor(Y_numpy, dtype=torch.float64, device=device)
b = torch.as_tensor(beta_numpy, dtype=torch.float64, device=device)
theta = torch.as_tensor(projections_numpy, dtype=torch.float64, device=device)

lm = solve_lmot(X, a, Y, b, projections=theta, device=device)
es = solve_est(X, a, Y, b, projections=theta, device=device)
sk = solve_sinkhorn(X, a, Y, b, epsilon=0.01, backend="pot", device=device)

P_lm = lm.plan.to_dense(max_entries=1_048_576)
cost = lm.squared_cost             # scalar torch tensor on CUDA
T = lm.barycentric_map             # torch tensor on CUDA
```

Both public import paths select the GPU backend. All three solvers use
`float64`, strictly positive weights summing to one, and distinct
atoms within each measure. Merge duplicate atoms and sum weights before solving.
Do not first round coordinates to float32 if exact equality must be preserved.
This implementation is for forward computation and benchmarking; it does not
add differentiable sorting or a differentiable transport layer.

## Mathematical rule and implementation scope

For each active projected block, LMOT still uses

```
H = equality_mask * min(u, v)
r = u - H.sum(axis=1)
t = v - H.sum(axis=0)
eta = 1 - H.sum()
Q = H + outer(r, t) / eta          # eta > mass_tol
Q = H                             # otherwise
```

The mask is exact equality of **all ground-space coordinates**, found once per
solve by lexicographically sorting the target and binary searching the source
rows. The search uses tensor-valued lower/upper bounds and comparisons on the
GPU; it does not construct a dense cost matrix or use approximate hashes. EST
does not perform overlap search. Both use the same quantile-block construction.

As before, fiber grouping uses anchored tolerance 1e-12, and positive residuals
at or below mass_tol=1e-12 are dropped. Marginal errors are measured explicitly.
Rare chains of near ties and nearly exhausted residual blocks use the same
fallback rules as the CPU implementation. Their scalar control decisions can
synchronize CUDA, while arrays and arithmetic remain on the device. CUDA
parallel reductions may differ from NumPy in the last floating-point bits.

## Validation

Cleanup validation used PyTorch 2.6.0+cpu:

- 12 of the 13 tests passed (9 backend tests and 3 simulation tests).
  The POT/Sinkhorn test was blocked by a bus error when importing the installed
  POT native extension `ot.bsp.bsp_wrap`; it remains enabled for Colab validation.
- The backend tests include 12 data settings x 4 directions x 2 lifted
  methods against an independent NumPy dense oracle; block mass, full plan,
  apply, cost, barycentric map and marginals were checked.
- Exact overlap tests include tied leading coordinates, signed zero and distinct
  atoms separated by 1e-14. Other cases cover permuted identity, full collapse,
  unequal support sizes, no overlap, singleton fibers, anchored near-tie chains,
  small residual mass and coordinates offset by 1e8. Public API routing to
  the CUDA-default backend is checked explicitly.
- The suite includes an analytic two-atom entropic check for POT and `torch_log`.
  The runner checks that passed use `torch_log` and cover synchronization ordering, CSV output,
  reference failure handling and implicit operation above the dense cap.
- The original smoke config completed through the torch runner in explicit CPU
  mode: 36 pair/method rows and six converged references at epsilon=0.01.
  LMOT's largest identity-plan entry error was about 6.25e-17.

**No CUDA hardware was available in the development workspace. Actual CUDA
execution and GPU speedups have not been measured here.** The Colab command
with `LMOT_REQUIRE_CUDA=1` runs the same oracle tests on CUDA in addition to CPU.
Run it before collecting GPU results for the paper. Small smoke problems and
float64 kernel-launch overhead need not be faster on a GPU.

Implementation references: [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html),
[stable argsort](https://docs.pytorch.org/docs/stable/generated/torch.argsort.html),
[POT log-domain Sinkhorn](https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_log).
