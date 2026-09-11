# Local validation

- Editable installation succeeded in a Python 3.12 environment.
- **13 unit tests passed**, including POT versus independent SciPy Sinkhorn.
- Independent LMOT/EST oracle covers 20 random measure pairs with 5 projections
  each, both methods, full coupling, block masses, marginals, entries, feature
  action, cost and maps; additional tests cover identity and numerical edge cases.
- Smoke simulation: **36 result rows**, status counts `{'ok': 30, 'not_converged': 6}`.
- Maximum LMOT/EST marginal L1 error in smoke: `9.194e-16`.
- Maximum LMOT/EST cost difference on random directions: `0.000e+00`.
- LMOT self-cost is zero up to floating-point roundoff.
- Both random and structured figures were generated; figure layout was inspected.
- A separate small d=3 common-support weight sweep ran, including the explicit
  `skipped_size` reporting path for Sinkhorn.
- Notebook code cells parse successfully; the same imported simulation entry
  point was executed. An actual Google Colab session was not used here.
- Full pilot/scaling sweeps have **not** been run. GitHub Actions is configured,
  but has not been executed on GitHub in this workspace.

The complete POT smoke outputs, config, source hash, versions and inputs are in
`examples/smoke_results/`. Sinkhorn nonconvergence at the smoke iteration budget
is retained as a result, not silently treated as a converged plan.
