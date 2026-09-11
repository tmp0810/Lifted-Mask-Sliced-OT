# LMOT: Lifted Mask Optimal Transport

Research code for **LMOT with exact binary overlap search**, **EST product
lifting**, and **balanced Sinkhorn**, with a reproducible simulation experiment.
The current scope is finite discrete measures with squared Euclidean ground cost.
MNIST interpolation and color transfer folders are reserved for later experiments.

All benchmarked transport methods use the PyTorch backend in `src/lmot/gpu/`, with CUDA as
the default device. Install with `pip install -e .`. See
[README_GPU.md](README_GPU.md) for Colab setup, tests and experiment commands.

Dense-plan RMSE uses unregularized OT linear programming (`POT ot.emd`) as
the reference. The LP runs on CPU outside method timing; LMOT, EST and
Sinkhorn predictions remain on the selected GPU.
