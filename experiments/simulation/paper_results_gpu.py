"""GPU-plan runtime + RMSE against OT linear programming + identity table.

LMOT, EST and Sinkhorn are predictions. POT ot.emd supplies a CPU reference
outside the prediction timer; existing data/projection YAML settings still apply.

    python -m experiments.simulation.paper_results_gpu \
        --config experiments/simulation/configs/smoke.yaml --device cuda
"""
from .run_gpu import main

if __name__ == "__main__":
    main(default_mode="dense")
