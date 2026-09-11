import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from lmot import solve_lmot, solve_est, solve_sinkhorn
from lmot.common import SizeLimitError
from lmot.metrics import exact_ot_cost
from lmot.projections import make_projections
from tests.reference_lmot import dense_reference


class LiftingTests(unittest.TestCase):
    def test_independent_reference(self):
        rng = np.random.default_rng(7)
        grid = np.array([(i, j) for i in range(5) for j in range(5)], float)
        for trial in range(20):
            x = grid[rng.choice(len(grid), 8, replace=False)]
            y = grid[rng.choice(len(grid), 11, replace=False)]
            a, b = rng.random(8)+.1, rng.random(11)+.1
            a, b = a/a.sum(), b/b.sum()
            theta = np.vstack(([[1, 0], [0, 1]], rng.normal(size=(3, 2))))
            for solver, masked in ((solve_lmot, True), (solve_est, False)):
                result = solver(x, a, y, b, projections=theta)
                full = result.plan.to_dense(batch_size=3)
                expected = sum(dense_reference(x, a, y, b, t, masked=masked)[0] for t in theta)/len(theta)
                assert_allclose(full, expected, atol=1e-12, rtol=1e-11)
                assert_allclose(full.sum(1), a, atol=1e-12)
                assert_allclose(full.sum(0), b, atol=1e-12)
                f = rng.normal(size=(len(y), 4))
                assert_allclose(result.plan.apply(f), expected @ f, atol=1e-12)
                C = np.sum((x[:, None]-y[None])**2, axis=2)
                assert_allclose(result.squared_cost, np.sum(expected*C), atol=1e-10)
                assert_allclose(result.barycentric_map, expected@y/a[:, None], atol=1e-11)
                ii, jj = np.indices(full.shape)
                assert_allclose(result.plan.entries(ii.ravel(), jj.ravel()), full.ravel(), atol=1e-12)
                row, col = result.plan.marginals()
                assert_allclose(row, full.sum(1), atol=1e-12)
                assert_allclose(col, full.sum(0), atol=1e-12)
                for t, part in zip(theta, result.plan.iter_slices()):
                    ref, blocks = dense_reference(x, a, y, b, t, masked=masked)
                    actual = part.apply(np.eye(len(y)))
                    for I, J, mass in blocks:
                        assert_allclose(actual[np.ix_(I, J)].sum(), mass, atol=1e-12)

    def test_identity_full_collapse_and_permutation(self):
        x = np.c_[np.zeros(16), np.arange(16)]
        a = np.arange(1, 17, dtype=float); a /= a.sum()
        p = np.random.default_rng(2).permutation(16)
        result = solve_lmot(x, a, x[p], a[p], projections=[[1., 0.]])
        expected = a[:, None]*np.all(x[:, None] == x[p][None], axis=2)
        assert_allclose(result.plan.to_dense(), expected, atol=1e-12)
        assert_allclose(result.squared_cost, 0, atol=1e-12)
        assert_allclose(result.barycentric_map, x, atol=1e-12)
        old = solve_est(x, a, x[p], a[p], projections=[[1., 0.]])
        self.assertGreater(old.squared_cost, 0)

    def test_est_skips_overlap(self):
        x = np.array([[0.,0.], [0.,1.]])
        with patch("lmot.methods.lmot.find_exact_overlap", side_effect=AssertionError("overlap called")):
            solve_est(x, [.5,.5], x, [.5,.5], projections=[[1.,0.]])

    def test_no_overlap_equals_est_and_singletons_equal(self):
        x = np.array([[0.,0.], [0.,1.], [1.,1.]])
        a = np.array([.2,.3,.5])
        for y, theta in ((x+.25, [[1.,0.]]), (x[::-1], [[1.,np.sqrt(2)]])):
            b = a[::-1]
            lm = solve_lmot(x,a,y,b,projections=theta)
            est = solve_est(x,a,y,b,projections=theta)
            assert_allclose(lm.plan.to_dense(), est.plan.to_dense(), atol=1e-12)

    def test_small_residual_large_offset(self):
        x = np.array([[0.,0.], [0.,2.]])+1e8
        a, b = np.array([.5,.5]), np.array([.5-1e-10,.5+1e-10])
        actual = solve_lmot(x,a,x,b,projections=[[1.,0.]])
        expected, _ = dense_reference(x,a,x,b,[1.,0.])
        C = np.sum((x[:,None]-x[None])**2,axis=2)
        assert_allclose(actual.plan.to_dense(), expected, atol=1e-12)
        assert_allclose(actual.squared_cost, np.sum(expected*C), atol=1e-14)

    def test_invalid_inputs_and_dense_guard(self):
        x = np.array([[0.,0.],[0.,1.]])
        with self.assertRaises(ValueError):
            solve_lmot(x,[.5,.5],x,[.5,.5],projections=[[0.,0.]])
        with self.assertRaises(ValueError):
            solve_lmot(x[[0,0]],[.5,.5],x,[.5,.5],projections=[[1.,0.]])
        result = solve_lmot(x,[.5,.5],x,[.5,.5],projections=[[1.,0.]])
        with self.assertRaises(SizeLimitError):
            result.plan.to_dense(max_entries=1)


class SinkhornTests(unittest.TestCase):
    def test_analytic_two_atom_solution(self):
        x = np.array([[0.,0.],[1.,0.]])
        eps = .2
        result = solve_sinkhorn(x,[.5,.5],x,[.5,.5],epsilon=eps)
        k = np.exp(-1/eps)
        expected = np.array([[1., k],[k, 1.]])/(2*(1+k))
        assert_allclose(result.plan.to_dense(), expected, atol=1e-12)
        assert_allclose(result.squared_cost, k/(1+k), atol=1e-12)
        self.assertTrue(result.diagnostics["converged"])

    def test_small_epsilon_and_exact_reference(self):
        x = np.array([[0.,0.],[1.,0.]])
        a, b = np.array([.7,.3]), np.array([.2,.8])
        result = solve_sinkhorn(x,a,x,b,epsilon=.02,max_iter=3000,tolerance=1e-10)
        cost = exact_ot_cost(x,a,x,b)
        assert_allclose(cost,.5,atol=1e-10)
        assert_allclose(result.squared_cost,cost,atol=1e-8)
        self.assertTrue(result.diagnostics["converged"])

    def test_nonconvergence_and_size_limit_are_visible(self):
        x = np.array([[0.,0.],[1.,0.]])
        result = solve_sinkhorn(x,[.9,.1],x,[.1,.9],epsilon=.001,max_iter=1)
        self.assertFalse(result.diagnostics["converged"])
        self.assertEqual(result.diagnostics["status"],"not_converged")
        with self.assertRaises(SizeLimitError):
            solve_sinkhorn(x,[.5,.5],x,[.5,.5],max_entries=1)

    @unittest.skipUnless(importlib.util.find_spec("ot"), "POT not installed")
    def test_pot_agreement(self):
        x = np.array([[0.,0.],[1.,0.]])
        for backend in ("scipy_log","pot"):
            r = solve_sinkhorn(x,[.7,.3],x,[.2,.8],epsilon=.1,backend=backend)
            if backend == "scipy_log":
                expected = r.plan.to_dense()
            else:
                assert_allclose(r.plan.to_dense(),expected,atol=1e-7)


if __name__ == "__main__":
    unittest.main()
