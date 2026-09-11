# Validation

The simulation's dense plan RMSE now uses unregularized OT linear programming
via POT `ot.emd`. LMOT, EST and Sinkhorn remain the benchmarked torch methods.
The reference runs on CPU outside prediction timing and is copied to the
prediction device only after validation. Data/projection generation, method
implementations, identity metrics and YAML configurations were not changed.

## Executed checks

- **18 tests passed**: 10 torch backend/runner tests, 5 LP-reference tests and
  3 simulation-data tests.
- The LP tests include coordinate-permuted self-transport with an exact identity
  plan, a known unequal-support optimal plan and a partial-overlap problem where
  forcing retention of common mass would be suboptimal.
- Feasible but unfinished or nonoptimal reference plans are rejected; an invalid
  reference produces no plan RMSE. Valid identity metrics remain available.
- Runner integration verifies one LP reference per dataset shared across all
  methods/direction budgets/repetitions, all three prediction methods, separate
  identity output, unchanged timing boundaries and no LP in implicit mode.
- Both CLI entry points and Colab code syntax were checked. The removed
  `--reference-epsilon` option is absent; LP budget/validation options remain.
- A representative n=4096, d=2, nonuniform, 50%-overlap reference (seed 0) solved
  with POT optimal status using the default 1,000,000-iteration budget. Marginal
  L1 errors were 5.13e-16 and 5.42e-16; primal/dual gap was 1.53e-12 and maximum
  dual-constraint violation 7.89e-11, within the 1e-9 validation tolerance.

Tests used PyTorch **2.6.0+cpu**, POT **0.9.6.post1**, NumPy **2.3.5** and
SciPy **1.17.0**. The local POT 0.9.7.post1 native import failed; validation was
completed with the compatible 0.9.6.post1 release. This was a local environment
change, not a change to the baseline implementation or project dependency range.

**CUDA hardware was unavailable here.** These checks exercise the torch
implementation on explicit CPU; they are not GPU runtime results. On Colab,
`LMOT_REQUIRE_CUDA=1 python -m unittest discover -s tests -v` also requires and
checks CUDA. The LP reference itself always runs on CPU.

`examples/smoke_results/` contains historical NumPy/CPU output. Older entropic
reference results must not be interpreted as results from the new LP protocol.
An OT LP may have multiple minimizers: RMSE compares with the particular plan
returned by POT; it is not a distance to the entire set of optimal couplings.
