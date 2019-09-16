import unittest
import torch
from analysis.utils import extract_weights
from analysis.path_selection import *
from torch import tensor


class TestPathSelection(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_top_n_weights(self):
        test_w = tensor([[[5., 3., 2., 2., 2.3, 5.],
                          [1., 0.3, 3., 3.3, 6., 5.],
                          [9., 4.5, 3., 7.6, 1., 2.]],
                         [[5., 3.11, 2.1, 1.1, 2.1, 5.],
                          [1.1, 0.3, 3., 3.3, 6., 5.],
                          [9.1, 4.5, 3.1, 7.6, 1., 2.]]])
        t = largest_n(test_w, 5, True)
        ref = [[1, 1, 4],  [0, 2, 3], [1, 2, 3], [1, 2, 0], [0, 2, 0]]
        self.assertEqual(ref, t)

    def test_top_n_weights_neg(self):
        test_w = tensor([[[-5., -3., -2., -2., -2.3, -5.],
                          [-1., -0.3, -3., -3.3, -6., -5.],
                          [-9., -4.5, -3., -7.6, -1., -2.]],
                         [[-5., -3.11, -2.1, -1.1, -2.1, -5.],
                          [-1.1, -0.3, -3., -3.3, -6., -5.],
                          [-9.1, -4.5, -3.1, -7.6, -1., -2.]]])

        t = largest_n(test_w, 5, False)
        ref = [[0, 1, 4], [0, 2, 3], [1, 2, 3], [1, 2, 0], [0, 2, 0]]

        self.assertEqual(ref, t)

    def test_top_weights_path(self):
        test_w = [tensor([[5., 3., 2., 2., 2.3, 5.],
                          [1., 0.3, 3., 3.3, 6., 5.],
                          [9., 4.5, 3., 7.6, 1., 2.]]),
                  tensor([[5., 3.11, 2.1, 1.1, 2.1, 5.],
                          [1.1, 0.3, 3., 3.3, 6., 5.],
                          [9.1, 4.5, 3.1, 7.6, 1., 2.]])]
        ref = [[[0, 5], [1, 4], [1, 5], [2, 3], [2, 0]], [[0, 5], [1, 5], [1, 4], [2, 3], [2, 0]]]
        self.assertEqual(ref, top_weights(list(), test_w, 5))

    def test_relevant_neurons_selection(self):
        test_rel = [tensor([2., 0.3, 0.8, 0.3, 1.1, 0.1, 0.09]),
                    tensor([0.1, 0.2, 0.9, 0.4, 2., 0.85, 1.3, 0.65, 0.72, 1.11]),
                    tensor([1.78, 0.14, 0.75, 0.94, 0.37, 1.87, 0.28, 1.26, 0.69, 0.80, 1.9, 0.12])]
        ref = [[[2, 9], [2, 4], [2, 6], [0, 9], [0, 4], [0, 6], [4, 9], [4, 4], [4, 6]], [[9, 0], [9, 10], [9, 5], [4, 0], [4, 10], [4, 5], [6, 0], [6, 10], [6, 5]]]
        self.assertEqual(ref, top_relevant_neurons(test_rel, list(), 3))

    def test_all_weights_linear(self):
        a = [[[2], [0], [4]], [[9], [4], [6]], [[0], [10], [5]]]
        ref = [[2, 9], [2, 4], [2, 6], [0, 9], [0, 4], [0, 6], [4, 9], [4, 4], [4, 6]]
        self.assertEqual(ref, all_weights(a[0], a[1]))

    def test_band(self):
        test_rel = [tensor([2., 0.3, 0.8, 0.3, 1.1, 0.1, 0.09]),
                    tensor([0.1, 0.2, 0.9, 0.4, 2., 0.85, 1.3, 0.65, 0.72, 1.11]),
                    tensor([1.78, 0.14, 0.75, 0.94, 0.37, 1.87, 0.28, 1.26, 0.69, 0.80, 1.9, 0.12])]
        ref = [[[0, 4], [0, 4], [0, 6], [0, 6]], [[4, 0], [4, 5], [6, 7], [6, 10]]]
        self.assertEqual(ref, band_selection(test_rel, list(), 2))

    def test_extract_weights(self):
        test_w = [tensor([[5., 3., 2., 2., 2.3, 5.],
                          [1., 0.3, 3., 3.3, 6., 5.],
                          [9., 4.5, 3., 7.6, 1., 2.]]),
                  tensor([[5., 3.11, 2.1, 1.1, 2.1, 5.],
                          [1.1, 0.3, 3., 3.3, 6., 5.],
                          [9.1, 4.5, 3.1, 7.6, 1., 2.]])]
        indices = [[[0, 5], [1, 4], [1, 5], [2, 3], [2, 0]],
                   [[0, 5], [1, 5], [1, 4], [2, 3], [2, 0]]]
        ref1 = tensor([5.0000, 6.0000, 5.0000, 7.6000, 9.0000, 5.0000, 5.0000, 6.0000, 7.6000,
                       9.1000])
        self.assertTrue(torch.allclose(ref1, extract_weights(test_w, indices)))
