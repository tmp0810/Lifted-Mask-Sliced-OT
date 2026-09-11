# Simulation

Run from the repository root after `pip install -e .`.

| YAML | Use |
|---|---|
| `configs/smoke.yaml` | Small complete run, including exact OT and figures |
| `configs/pilot.yaml` | Support-overlap sweep, two weight modes, d=2/3, multiple L |
| `configs/weights.yaml` | Common support with changing weights |
| `configs/scaling.yaml` | Larger N; explicit dense Sinkhorn size limits |

Example: `python -m experiments.simulation.run --config experiments/simulation/configs/pilot.yaml`.
Outputs go into a fresh `results/simulation/<name>_<timestamp>/` folder. Override
with `--output-dir NEW_FOLDER`. Use `--no-plots` to skip figure rendering.

The two sliced methods receive the same saved directions for each configuration.
Sinkhorn is evaluated once per dataset and epsilon, independently of L/direction
kind. Compare only rows with compatible dataset, output, hardware and status.

Change `geometry: continuous` to test continuous point clouds while preserving
the controlled overlap. Change `projection_kinds` to `[random]` or `[structured]`
for separate sweeps. `overlap_values` applies to `scenario: support`;
`weight_change_values` applies to `scenario: weights`.

Notebook entry point: `notebooks/simulation_colab.ipynb`.
