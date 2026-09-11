"""Full GPU-plan runtime + RMSE + separate identity table; existing YAML configs.

    python -m experiments.simulation.paper_results_gpu \
        --config experiments/simulation/configs/smoke.yaml --device cuda
"""
from .run_gpu import main

if __name__ == "__main__":
    main(default_mode="dense")
