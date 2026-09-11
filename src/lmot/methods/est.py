"""Reimplementation of EST's general-discrete product lifting rule."""
import numpy as np
from ..common import prepare_measures, prepare_projections
from ..plans import SlicedPlan


def solve_est(x, alpha, y, beta, *, projections, fiber_tol=1e-12, mass_tol=1e-12):
    """Uniform projection average. No overlap index is constructed or queried."""
    x, alpha, y, beta = prepare_measures(x, alpha, y, beta)
    theta = prepare_projections(projections, x.shape[1])
    empty = np.empty(0, dtype=np.int64)
    return SlicedPlan(x, alpha, y, beta, theta, empty, empty, method="EST",
                      fiber_tol=fiber_tol, mass_tol=mass_tol).evaluate()
