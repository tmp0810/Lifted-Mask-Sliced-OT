# Simulation (CUDA)

Run from the repository root after `pip install -e .`.

| YAML | Use |
|---|---|
| `configs/smoke.yaml` | Small run to check dense-plan output, RMSE and identity |
| `configs/pilot.yaml` | Main support-overlap sweep: d=2, nonuniform weights, structured directions, varying n and L |
| `configs/weights.yaml` | Common support with changing weights |
| `configs/scaling.yaml` | Larger n; implicit lifting and explicit dense Sinkhorn limits |

Dense-plan runtime, RMSE against an OT linear-programming reference, and separate identity checks:

```bash
python -m experiments.simulation.paper_results_gpu --config experiments/simulation/configs/pilot.yaml --device cuda --max-entries 16777216
```

Implicit cost/map runtime for larger problems:

```bash
python -m experiments.simulation.run_gpu --config experiments/simulation/configs/scaling.yaml --device cuda
```

These commands use the same GPU implementations; `output_mode` records the
output protocol. They require CUDA unless `--device cpu` is explicitly selected
for a correctness check. The legacy `run.py`, `paper_results.py` and CPU CSV
plotter have been removed.

Dense output includes `paper_results.csv/.tsv`, `identity_results.csv/.tsv`,
`per_pair.csv`, `raw_timings.csv`, configuration and metadata. Implicit output
uses `summary.csv/.tsv`, with no dense plan RMSE. A fresh result directory is
created for each run; use `--output-dir NEW_FOLDER` to choose it.

The two sliced methods receive the same saved directions for each configuration.
Sinkhorn is evaluated once per dataset and epsilon, independently of L/direction
kind. Compare only rows with compatible dataset, output, hardware and status.

The effective dense cap is the smaller of `--max-entries` and the YAML's
`sinkhorn.max_entries`. Both must permit n*m entries to run a dense case.
The reference is always unregularized OT solved by POT's `ot.emd` on CPU,
then transferred to the prediction device outside timing. LMOT, EST and
Sinkhorn remain the three benchmarked methods. The YAML's `sinkhorn.epsilons`
selects baseline epsilons; the reference has no epsilon. Remove the obsolete
`--reference-epsilon` option from older commands.

`--reference-max-iter` (default 1,000,000) sets the network-simplex iteration
budget. `--reference-tolerance` (default 1e-9) checks marginals and LP optimality.
Failed references produce blank RMSE and an explicit status, never a fallback
reference. Detailed output includes the LP cost and validation diagnostics;
main tables identify `reference_method=ot_lp` and `reference_backend=pot.emd`.

Legacy `exact_max_entries` config fields remain accepted but unused; the two
current dense caps above govern the LP reference as well as dense predictions.
For n=4096, both must allow at least 16,777,216 entries. LP reference storage and
computation require CPU resources even when predictions run on CUDA.

An LP may have multiple optimal couplings; plan RMSE compares with the
particular minimizer returned by POT. Identity checks still use the exact
coordinate-aligned self-coupling.

Change `geometry: continuous` to test continuous point clouds while preserving
controlled overlap. `overlap_values` applies to `scenario: support`;
`weight_change_values` applies to `scenario: weights`.

`utils.py` handles configuration, provenance and tables; `data.py` generates
the measures. `ot_reference.py` is the untimed LP evaluator, separate from the
production GPU methods.

Notebook: `notebooks/simulation_colab.ipynb`. Full timing conventions and
failure handling: [README_GPU.md](../../README_GPU.md).
