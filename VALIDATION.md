# Validation

The production code now has one PyTorch transport backend in `src/lmot/gpu/`.
The public `lmot` API selects it with CUDA by default. Dataset generation,
direction banks and CSV I/O remain host utilities.

- Current suite has 13 tests. **12 passed** during this cleanup (9 backend tests
  and 3 simulation tests). The POT/Sinkhorn test could not complete: importing
  the installed POT native extension `ot.bsp.bsp_wrap` caused a process-level
  bus error. The test remains enabled in the repository.
- Tests use the same torch implementation on explicit CPU and also on CUDA when
  available. `LMOT_REQUIRE_CUDA=1` fails if CUDA is absent.
- The independent NumPy oracle in `tests/reference_lmot.py` checks the lifting
  mathematics; it is not imported by the production solver or runner.
- Checks cover full plans, block masses, marginals, actions, costs, barycentric
  maps, overlap search, collision identity, unequal supports, near-tie grouping,
  tiny residuals and large coordinate offsets.
- The suite includes a two-atom entropic check of POT and `torch_log`. The
  runner checks that passed use `torch_log` references and cover
  dense and implicit modes, synchronization, reference failure and size limits.
- Simulation tests verify nested overlap, weight changes and projection prefixes.
- Both GPU entry points and the Colab notebook refer only to the current modules.

Validation here used PyTorch 2.6.0+cpu. **CUDA hardware was not available in this
workspace**, so no new CUDA execution or GPU speedup is claimed by these checks.
The original torch smoke validation (before removing the duplicate CPU source)
completed 36 pair/method rows and six converged references at epsilon=0.01;
LMOT identity maximum entry error was about 6.25e-17.

`examples/smoke_results/` contains older NumPy/CPU output preserved as historical
data. It is not a current GPU benchmark or evidence that the CUDA tests ran.
