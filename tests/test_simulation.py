import unittest
import numpy as np
import torch
from numpy.testing import assert_array_equal, assert_allclose
from experiments.simulation.data import make_pair
from lmot.gpu.metrics import overlap_statistics, collision_statistics
from lmot.projections import make_projections


def as_tensors(args):
    """The production torch backend on explicit CPU for dataset unit tests."""
    return tuple(torch.as_tensor(v, dtype=torch.float64) for v in args)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_overlap_sweep_is_nested_and_scaled(self):
        previous = None
        for rho in (1., .75, .5, .25, 0.):
            args, self_case = make_pair(32, 3, value=rho, seed=7)
            x, a, y, b = args
            i, j, stats = overlap_statistics(*as_tensors(args))
            self.assertEqual(len(i), round(32*rho))
            shared = set(map(tuple, x[i.numpy()]))
            if previous is not None:
                self.assertTrue(shared <= previous)
            previous = shared
            self.assertEqual(self_case, rho == 1)
            self.assertLessEqual(np.max(np.sum((x[:,None]-y[None])**2,axis=2)),1+1e-12)
            assert_allclose(a.sum(),1)
            assert_allclose(b.sum(),1)
            again, _ = make_pair(32, 3, value=rho, seed=7)
            for left, right in zip(args, again):
                assert_array_equal(left,right)

    def test_common_support_changes_weights_only(self):
        start, _ = make_pair(16,2,value=0.,seed=0,scenario="weights")
        end, _ = make_pair(16,2,value=1.,seed=0,scenario="weights")
        assert_array_equal(start[0],end[0])
        assert_array_equal(start[2],end[2])
        self.assertFalse(np.allclose(start[3],end[3]))
        self.assertEqual(overlap_statistics(*as_tensors(end))[2]["overlap_count"],16)

    def test_shared_projection_prefixes_and_collision_control(self):
        args, _ = make_pair(32,2,value=.5,seed=0)
        for kind in ("random","structured"):
            small=make_projections(2,8,kind=kind,seed=19)
            large=make_projections(2,32,kind=kind,seed=19)
            assert_array_equal(small,large[:8])
            assert_allclose(np.linalg.norm(large,axis=1),1)
            collisions=collision_statistics(*as_tensors(args),torch.as_tensor(small, dtype=torch.float64))
            if kind == "random":
                self.assertEqual(collisions["collision_fraction"],0)
            else:
                self.assertGreater(collisions["collision_fraction"],0)
