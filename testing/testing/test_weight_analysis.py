import unittest

from analysis.weight_analysis import *
from torch import tensor


class TestWeightAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        self.final = tensor([[0.8, 0.2],
                             [-0.1, 0.7]])
        self.t = tensor([[0.5, 0.5],
                         [0.1, 0.5]])
        self.t_1 = tensor([[0.4, 0.45],
                           [0.15, 0.4]])

    def tearDown(self) -> None:
        pass

    def test_dist_to_final(self):
        ref1 = tensor([[0.3000, 0.3000],
                       [0.2000, 0.2000]])
        ref2 = tensor([[0.4000, 0.2500],
                       [0.2500, 0.3000]])
        self.assertTrue(torch.allclose(ref1, dist_to_final(self.t, self.final)))
        self.assertTrue(torch.allclose(ref2, dist_to_final(self.t_1, self.final)))

    def test_step_diff(self):
        ref = tensor([[-0.1000, 0.0500],
                      [-0.0500, -0.1000]])
        self.assertTrue(torch.allclose(ref, step_diff(self.t, self.t_1, self.final)))

    def test_avg_diff(self):
        ref = tensor(-0.0500)
        self.assertTrue(torch.allclose(ref, avg_diff(self.t, self.t_1, self.final)))

    def test_total_diff(self):
        ref = tensor(-0.2000)
        self.assertTrue(torch.allclose(ref, total_diff(self.t, self.t_1, self.final)))

    def test_pos_neg_diff(self):
        ref = tensor([0.0500, -0.2500])
        self.assertTrue(torch.allclose(ref, pos_neg_diff(self.t, self.t_1, self.final)))

