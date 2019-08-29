import unittest
from torch import tensor
from analysis.path_selection import *


class TestPathSelection(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_top_n_weights(self):
        test_w = torch.tensor([[[5., 3., 2., 2., 2.3, 5.],
                                [1., 0.3, 3., 3.3, 6., 5.],
                                [9., 4.5, 3., 7.6, 1., 2.]],
                               [[5., 3.11, 2.1, 1.1, 2.1, 5.],
                                [1.1, 0.3, 3., 3.3, 6., 5.],
                                [9.1, 4.5, 3.1, 7.6, 1., 2.]]])
        t = top_n_weights(test_w, 5)
        ref = [(tensor(6.), (1, 1, 4)), (tensor(7.6000), (0, 2, 3)), (tensor(7.6000), (1, 2, 3)), (tensor(9.1000), (1, 2, 0)), (tensor(9.), (0, 2, 0))]
        ans = 0.
        for i in range(len(t)):
            ans += t[i][0] - ref[i][0]
        self.assertTrue(-0.001 < ans < 0.001)

    def test_top_n_weights_neg(self):
        test_w = torch.tensor([[[-5., -3., -2., -2., -2.3, -5.],
                                [-1., -0.3, -3., -3.3, -6., -5.],
                                [-9., -4.5, -3., -7.6, -1., -2.]],
                               [[-5., -3.11, -2.1, -1.1, -2.1, -5.],
                                [-1.1, -0.3, -3., -3.3, -6., -5.],
                                [-9.1, -4.5, -3.1, -7.6, -1., -2.]]])
        dims = len(test_w.size())
        indices = [0] * dims

        t = recurse_large_neg(test_w, list(), indices, 5, 0)
        ref = [(tensor(-6.), (1, 1, 4)), (tensor(-7.6000), (0, 2, 3)), (tensor(-7.6000), (1, 2, 3)),
               (tensor(-9.1000), (1, 2, 0)), (tensor(-9.), (0, 2, 0))]
        ans = 0.
        for i in range(len(t)):
            ans += t[i][0] - ref[i][0]
        self.assertTrue(-0.001 < ans < 0.001)
