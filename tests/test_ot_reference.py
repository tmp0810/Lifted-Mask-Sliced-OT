"""Analytic transport cases and rejection of invalid LP ground truth."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.simulation.ot_reference import solve_ot_reference
from lmot.gpu.common import SizeLimitError


class TestOTReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    @staticmethod
    def tensors(values, device):
        return tuple(torch.as_tensor(v, dtype=torch.float64, device=device) for v in values)

    def test_permuted_self_transport_is_identity(self):
        x = np.array([[0., 0.], [1., 0.], [1., 2.]])
        a, perm = np.array([.2, .3, .5]), np.array([2, 0, 1])
        expected = np.diag(a)[:, perm]
        for device in self.devices:
            P, info = solve_ot_reference(*self.tensors((x, a, x[perm], a[perm]), device))
            self.assertEqual(info['reference_status'], 'ok')
            self.assertEqual(P.device.type, device)
            self.assertEqual(P.dtype, torch.float64)
            np.testing.assert_allclose(P.cpu(), expected, atol=1e-14, rtol=0)
            self.assertAlmostEqual(info['reference_cost'], 0., places=13)

    def test_unequal_supports_have_known_unregularized_plan(self):
        # Strictly convex 1D cost: the unique monotone plan splits source mass.
        values = ([[0., 0.], [3., 0.]], [.4, .6],
                  [[1., 0.], [2., 0.], [4., 0.]], [.2, .3, .5])
        expected = np.array([[.2, .2, 0.], [0., .1, .5]])
        for device in self.devices:
            P, info = solve_ot_reference(*self.tensors(values, device))
            self.assertEqual(info['reference_status'], 'ok')
            np.testing.assert_allclose(P.cpu(), expected, atol=1e-14, rtol=0)
            self.assertAlmostEqual(info['reference_cost'], 1.6, places=13)
            self.assertLess(info['reference_dual_gap'], 1e-12)
            self.assertLess(info['reference_dual_violation'], 1e-12)

    def test_partial_overlap_is_not_forced_to_preserve_common_mass(self):
        # Retaining the shared atom at 1 costs 2. The true OT optimum costs 1.
        values = ([[0., 0.], [1., 0.]], [.5, .5],
                  [[1., 0.], [2., 0.]], [.5, .5])
        P, info = solve_ot_reference(*self.tensors(values, 'cpu'))
        self.assertEqual(info['reference_status'], 'ok')
        np.testing.assert_allclose(P, np.diag([.5, .5]), atol=1e-14, rtol=0)
        self.assertAlmostEqual(info['reference_cost'], 1., places=13)

    def test_feasible_but_unfinished_or_nonoptimal_plans_are_rejected(self):
        args = self.tensors(([[0., 0.], [1., 0.]], [.5, .5],
                             [[0., 0.], [1., 0.]], [.5, .5]), 'cpu')
        # Row/column checks alone would accept this cost-one plan; optimum is zero.
        wrong = np.array([[0., .5], [.5, 0.]])
        for code, warning, status in ((3, 'iteration limit', 'not_optimal'),
                                      (1, None, 'invalid_optimality')):
            log = dict(result_code=code, warning=warning, u=np.zeros(2), v=np.zeros(2))
            with patch('ot.emd', return_value=(wrong, log)):
                P, info = solve_ot_reference(*args)
            self.assertIsNone(P)
            self.assertEqual(info['reference_status'], status)
            self.assertEqual(info['reference_row_l1'], 0.)
            self.assertEqual(info['reference_col_l1'], 0.)

    def test_size_and_budget_limits(self):
        args = self.tensors(([[0., 0.], [1., 0.]], [.5, .5],
                             [[0., 0.], [1., 0.]], [.5, .5]), 'cpu')
        with patch('ot.emd', side_effect=AssertionError('LP should not run')):
            with self.assertRaises(SizeLimitError):
                solve_ot_reference(*args, max_entries=3)
            with self.assertRaises(ValueError):
                solve_ot_reference(*args, max_iter=0)


if __name__ == '__main__':
    unittest.main()
