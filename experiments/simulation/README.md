# Simulation (CUDA)

Run from the repository root after `pip install -e .`.

| YAML | Use |
|---|---|
| `configs/smoke.yaml` | Small run to check dense-plan output, RMSE and identity |
| `configs/pilot.yaml` | Main support-overlap sweep: d=2, nonuniform weights, structured directions, varying n and L |
| `configs/weights.yaml` | Common support with changing weights |
| `configs/scaling.yaml` | Larger n; implicit lifting and explicit dense Sinkhorn limits |

Dense-plan runtime, RMSE against converged Sinkhorn, and separate identity checks:

```bash
python -m experiments.simulation.paper_results_gpu --config experiments/simulation/configs/pilot.yaml --device cuda
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
`--reference-epsilon` sets the RMSE reference (default 0.001); the YAML's
`sinkhorn.epsilons` selects baseline epsilons independently. Legacy
`exact_max_entries` config fields are accepted but unused: there is no CPU LP
reference solve in the GPU runner.

Change `geometry: continuous` to test continuous point clouds while preserving
controlled overlap. `overlap_values` applies to `scenario: support`;
`weight_change_values` applies to `scenario: weights`.

`utils.py` handles configuration, provenance and tables. `data.py` handles data
generation. Neither implements a CPU transport backend.

Notebook: `notebooks/simulation_colab.ipynb`. Full timing conventions and
failure handling: [README_GPU.md](../../README_GPU.md).
