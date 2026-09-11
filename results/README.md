Each GPU simulation creates a fresh directory with configuration, environment
metadata, projection banks, optional inputs, raw timings and summary CSV/TSV.
Dense mode adds plan RMSE and a separate identity table; implicit mode reports
cost/map runtime. Checkpoints are written after each dataset. DONE.json means
computation finished, not that every Sinkhorn call converged. Generated runs are
excluded from git; preserve selected artifacts separately for paper results.
