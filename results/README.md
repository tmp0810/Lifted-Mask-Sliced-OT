Each GPU simulation creates a fresh directory with configuration, environment
metadata, projection banks, optional inputs, raw timings and summary CSV/TSV.
Dense mode adds plan RMSE against POT unregularized OT linear programming and a separate identity table; implicit mode reports
cost/map runtime. Checkpoints are written after each dataset. DONE.json means
computation finished, not that every Sinkhorn call converged. Generated runs are
excluded from git; preserve selected artifacts separately for paper results.

Use `reference_method` and `reference_backend` to identify the reference.
LP solver status, cost, marginal errors and dual certificate are recorded in
`per_pair.csv`; reference construction/transfer never enters method runtime.
Old entropic-reference results must be reported separately from LP results.
