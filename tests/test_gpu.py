"""Run CPU parity and, when present, the SAME tests on CUDA.

    LMOT_REQUIRE_CUDA=1 python -m unittest tests.test_gpu -v

The environment flag makes missing CUDA a failure rather than a silent skip.
NumPy reference calculations are used only in this test module, never by GPU solvers.
"""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
try:
    import torch
except ImportError:
    torch = None

if torch is None and os.environ.get("LMOT_REQUIRE_CUDA") == "1":
    raise RuntimeError("LMOT_REQUIRE_CUDA=1 but torch is not installed")

from tests.reference_lmot import dense_reference
from experiments.simulation.data import make_pair


@unittest.skipIf(torch is None, "install the project dependencies to test the torch backend")
class TestTorchBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("LMOT_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
            raise RuntimeError("LMOT_REQUIRE_CUDA=1 but CUDA is unavailable")
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_binary_search_all_coordinates(self):
        from lmot.gpu.common import find_exact_overlap, tensor
        rng = np.random.default_rng(123)
        x = np.unique(rng.integers(-5, 6, (100, 3)), axis=0).astype(float)
        y = np.unique(np.vstack((x[::3], rng.integers(-8, 9, (60, 3)))), axis=0)
        # First-coordinate ties, negative values, signed zero and near-but-not-equal atoms.
        x = np.vstack((x, [[100, 0, 0], [100, 0, 1e-14]]))
        y = np.vstack((y, [[100, -0., 0]]))
        expected = set(zip(*np.nonzero(np.all(x[:, None, :] == y[None, :, :], axis=2))))
        for device in self.devices:
            i, j = find_exact_overlap(tensor(x, device), tensor(y, device))
            self.assertEqual(set(zip(i.cpu().tolist(), j.cpu().tolist())), expected)
            self.assertEqual(i.device.type, device)

    def test_independent_dense_oracle(self):
        from lmot.gpu import solve_lmot, solve_est
        from lmot.gpu.common import tensor
        directions = np.array([[1., 0.], [0., 1.], [1., 1.], [np.sqrt(2), -1.]])
        for seed in range(12):
            args, _ = make_pair(5 + seed, 2, value=(seed % 3) / 2, seed=seed,
                               weights="uniform" if seed % 2 else "nonuniform",
                               scenario="support" if seed < 6 else "weights")
            x, a, y, b = args
            features = np.random.default_rng(seed).normal(size=(len(y), 3))
            cost = ((x[:, None] - y[None, :])**2).sum(2)
            for device in self.devices:
                for name, solver in (("LMOT", solve_lmot), ("EST", solve_est)):
                    with self.subTest(seed=seed, device=device, method=name):
                        result = solver(*args, projections=directions, device=device)
                        dense = result.plan.to_dense(batch_size=3)
                        references = [dense_reference(*args, theta, masked=name == "LMOT")
                                      for theta in directions]
                        expected = sum(p for p, _ in references) / len(directions)
                        np.testing.assert_allclose(dense.cpu().numpy(), expected, atol=2e-13, rtol=0)
                        np.testing.assert_allclose(result.plan.apply(tensor(features, device)).cpu(),
                                                   expected @ features, atol=2e-13, rtol=0)
                        np.testing.assert_allclose(result.barycentric_map.cpu(),
                                                   expected @ y / a[:, None], atol=2e-12, rtol=0)
                        self.assertAlmostEqual(float(result.squared_cost), float((expected * cost).sum()), places=12)
                        row, col = result.plan.marginals()
                        np.testing.assert_allclose(row.cpu(), a, atol=2e-13, rtol=0)
                        np.testing.assert_allclose(col.cpu(), b, atol=2e-13, rtol=0)
                        self.assertEqual(dense.device.type, device)
                        for part, (oracle, blocks) in zip(result.plan.iter_slices(), references):
                            ii, jj = np.indices(oracle.shape)
                            got = part.entries(ii, jj).cpu().numpy()
                            np.testing.assert_allclose(got, oracle, atol=2e-13, rtol=0)
                            for I, J, mass in blocks:
                                self.assertAlmostEqual(float(got[np.ix_(I, J)].sum()), mass, places=12)

    def test_unequal_sizes_and_full_collapse(self):
        from lmot.gpu import solve_lmot, solve_est
        x = np.array([[0., -2.], [0., 0.], [0., 4.]])
        y = np.array([[0., 4.], [0., 1.], [0., 0.], [0., 5.]])
        a, b, theta = np.array([.1, .3, .6]), np.array([.5, .2, .1, .2]), np.array([[1., 0.]])
        for device in self.devices:
            for name, fn in (("LMOT", solve_lmot), ("EST", solve_est)):
                result = fn(x, a, y, b, projections=theta, device=device)
                expected, _ = dense_reference(x, a, y, b, theta[0], masked=name == "LMOT")
                np.testing.assert_allclose(result.plan.to_dense(batch_size=2).cpu(), expected, atol=1e-14)

    def test_identity_and_est_collapse(self):
        from lmot.gpu import solve_lmot, solve_est
        from lmot.gpu.metrics import identity_reference, overlap_statistics, plan_rmse
        from lmot.gpu.common import tensor
        x = np.column_stack((np.zeros(7), np.arange(7)))
        a = np.arange(1., 8.) / 28
        perm = np.array([4, 2, 0, 6, 1, 5, 3])
        for device in self.devices:
            args = tuple(tensor(v, device) for v in (x, a, x[perm], a[perm]))
            i, j, _ = overlap_statistics(*args)
            identity = identity_reference(*args, i, j)
            result = solve_lmot(*args, projections=[[1., 0.]], device=device)
            self.assertLess(plan_rmse(result.plan.to_dense(), identity), 1e-14)
            self.assertLess(abs(float(result.squared_cost)), 1e-13)
            np.testing.assert_allclose(result.barycentric_map.cpu(), x, atol=1e-12, rtol=0)
            est = solve_est(*args, projections=[[1., 0.]], device=device)
            self.assertGreater(plan_rmse(est.plan.to_dense(), identity), 1e-3)

    def test_no_overlap_and_singleton_equality(self):
        from lmot.gpu import solve_lmot, solve_est
        for device in self.devices:
            for value, theta in ((0., [[1., 0.]]), (.5, [[np.sqrt(2), 1.]])):
                args, _ = make_pair(16, 2, value=value, seed=3)
                lm = solve_lmot(*args, projections=theta, device=device).plan.to_dense()
                es = solve_est(*args, projections=theta, device=device).plan.to_dense()
                torch.testing.assert_close(lm, es, atol=1e-13, rtol=0)
            with patch("lmot.gpu.methods.find_exact_overlap", side_effect=AssertionError("EST searched overlap")):
                solve_est(*args, projections=theta, device=device)

    def test_anchored_tolerance_and_small_residual(self):
        from lmot.gpu.common import tensor, make_fibers
        from lmot.gpu import solve_lmot
        x = np.column_stack((np.array([0., .75, 1.5, 2.1, 3.]) * 1e-12, np.arange(5.)))
        a, theta = np.ones(5)/5, np.array([1., 0.])
        # Anchors 0, 1.5e-12 and 3e-12 give these exact groups.
        expected_groups = np.array([0, 0, 1, 1, 2])
        for device in self.devices:
            f = make_fibers(tensor(x, device), tensor(a, device), tensor(theta, device))
            np.testing.assert_array_equal(f.groups.cpu(), expected_groups)
            xx = np.array([[1e8, 1e8], [1e8, 1e8 + 1], [1e8, 1e8 + 3]])
            aa = np.array([.2, .3, .5])
            bb = aa + np.array([1e-10, -1e-10, 0])
            p, _ = dense_reference(xx, aa, xx, bb, theta)
            result = solve_lmot(xx, aa, xx, bb, projections=theta[None, :], device=device)
            np.testing.assert_allclose(result.plan.to_dense().cpu(), p, atol=1e-13, rtol=0)
            cost = ((xx[:, None] - xx[None, :])**2).sum(2)
            self.assertAlmostEqual(float(result.squared_cost), float((p * cost).sum()), places=12)

    def test_sinkhorn_and_plan_rmse(self):
        from lmot.gpu import solve_sinkhorn
        from lmot.gpu.common import tensor
        from lmot.gpu.metrics import plan_rmse
        x, a = np.array([[0., 0.], [1., 0.]]), np.array([.5, .5])
        eps = .2
        off = .5 * np.exp(-1/eps) / (1 + np.exp(-1/eps))
        expected = np.array([[.5-off, off], [off, .5-off]])
        for device in self.devices:
            for backend in ("pot", "torch_log"):
                r = solve_sinkhorn(x, a, x, a, device=device, backend=backend, epsilon=eps, tolerance=1e-10)
                self.assertEqual(r.diagnostics["status"], "ok")
                np.testing.assert_allclose(r.plan.matrix.cpu(), expected, atol=1e-10, rtol=0)
                self.assertEqual(r.plan.matrix.device.type, device)
                self.assertAlmostEqual(plan_rmse(r.plan.matrix, tensor(np.diag(a), device)), off, places=10)
                failed = solve_sinkhorn(x, [.9, .1], x, [.1, .9], device=device, backend=backend,
                                        epsilon=.01, max_iter=1, tolerance=1e-12)
                self.assertEqual(failed.diagnostics["status"], "not_converged")

    def test_validation_and_limits(self):
        from lmot.gpu import solve_lmot, solve_sinkhorn, resolve_device
        from lmot.gpu.common import SizeLimitError
        args, _ = make_pair(4, 2, value=.5, seed=0)
        for device in self.devices:
            result = solve_lmot(*args, projections=[[1., 0.]], device=device)
            with self.assertRaises(SizeLimitError):
                result.plan.to_dense(max_entries=15)
            with self.assertRaises(SizeLimitError):
                solve_sinkhorn(*args, device=device, max_entries=15)
            with self.assertRaises(ValueError):
                solve_lmot([[0., 0.], [0., 0.]], [.5, .5], [[1., 1.]], [1.],
                           projections=[[1., 0.]], device=device)
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_runner_outputs_and_timing_synchronization(self):
        from experiments.simulation.run_gpu import run, timed_prediction, GROUP_COLUMNS, aggregate
        events = []
        with patch("experiments.simulation.run_gpu.synchronize", side_effect=lambda d: events.append("sync")), \
             patch("experiments.simulation.run_gpu.prediction", side_effect=lambda *a: (events.append("predict"), None)):
            timed_prediction(None, device=torch.device("cpu"), output_mode="dense", max_entries=4, batch_size=2)
        self.assertEqual(events, ["sync", "predict", "sync"])
        base = dict(zip(GROUP_COLUMNS, ["support", "grid", 4, 4, 2, "uniform", .5,
                                        "random", 1, "LMOT", None, .01, "cpu", "float64", "dense", "torch"]))
        records = [dict(base, is_self=False, rmse_status="ok", runtime_ms=t, plan_rmse=e)
                   for t, e in [(2., .01), (4., .03)]]
        table = aggregate(records)[0]
        self.assertAlmostEqual(table['runtime_ms_std'], np.sqrt(2))
        self.assertAlmostEqual(table['plan_rmse_mean'], .02)
        config = yaml.safe_load((Path(__file__).parents[1]/"experiments/simulation/configs/smoke.yaml").read_text())
        config.update(sizes=[4], seeds=[0], overlap_values=[1., .5, 0.], projection_counts=[2],
                      repeats=1, warmups=1, save_inputs=False, measure_memory=False)
        config['sinkhorn'].update(backend="torch_log", epsilons=[.1], max_iter=1000)
        for device in self.devices:
            with tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                cfg = folder/'check.yaml'
                cfg.write_text(yaml.safe_dump(config))
                out = run(cfg, folder/'dense', device=device, output_mode='dense', reference_epsilon=.1)
                rows = list(csv.DictReader((out/'per_pair.csv').read_text().splitlines()))
                self.assertEqual(len(rows), 15)
                self.assertTrue(all(r['reference_status']=='ok' for r in rows))
                self.assertTrue(all(r['rmse_status']=='ok' for r in rows))
                self.assertTrue(all(r['device'].startswith(device) for r in rows))
                for row in rows:
                    if row['method']=='LMOT' and row['is_self']=='True':
                        self.assertLess(float(row['identity_rmse']), 1e-13)
                # Implicit mode keeps lifted methods above the dense cap.
                out = run(cfg, folder/'implicit', device=device, output_mode='implicit', max_entries=1)
                rows = list(csv.DictReader((out/'per_pair.csv').read_text().splitlines()))
                self.assertTrue(all(r['plan_status']=='skipped_size' for r in rows if r['method']=='Sinkhorn'))
                self.assertTrue(all(r['plan_status']=='ok' for r in rows if r['method']!='Sinkhorn'))
                # A failed reference must not yield a purported ground-truth RMSE.
                out = run(cfg, folder/'failed_ref', device=device, output_mode='dense',
                          reference_epsilon=.01, reference_max_iter=1)
                rows = list(csv.DictReader((out/'per_pair.csv').read_text().splitlines()))
                failed = [r for r in rows if r['reference_status']!='ok']
                self.assertTrue(failed)
                self.assertTrue(all(not r['plan_rmse'] for r in failed))

    def test_public_api_selects_gpu_backend(self):
        import lmot
        import lmot.gpu
        for name in ("solve_lmot", "solve_est", "solve_sinkhorn"):
            self.assertIs(getattr(lmot, name), getattr(lmot.gpu, name))
        args, _ = make_pair(4, 2, value=.5, seed=0)
        # Default public calls must require CUDA; never silently use NumPy/CPU.
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                lmot.solve_lmot(*args, projections=[[1., 0.]])


if __name__ == "__main__":
    if torch is None and os.environ.get("LMOT_REQUIRE_CUDA") == "1":
        raise RuntimeError("torch is not installed")
    unittest.main()
