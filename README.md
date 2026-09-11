# LMOT: Lifted Mask Optimal Transport

Research code for **LMOT with exact binary overlap search**, **EST product
lifting**, and **balanced Sinkhorn**, with a reproducible simulation experiment.
The current scope is finite discrete measures with squared Euclidean ground cost.
MNIST interpolation and color transfer folders are reserved for later experiments.


## GPU backend (PyTorch / CUDA)

LMOT, EST and Sinkhorn also have a float64 PyTorch backend. See
[README_GPU.md](README_GPU.md) for Colab setup, CUDA correctness checks,
timing conventions, and dense-plan RMSE versus implicit scaling experiments.

```bash
python -m pip install -e ".[gpu]"
LMOT_REQUIRE_CUDA=1 python -m unittest tests.test_gpu -v
python -m experiments.simulation.paper_results_gpu --config experiments/simulation/configs/smoke.yaml --device cuda
```

The existing commands select the NumPy/CPU backend. Use the `_gpu` entry points
to select CUDA; they do not silently fall back to CPU.
