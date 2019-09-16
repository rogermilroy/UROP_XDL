import unittest

import torch
from analysis.weight_analysis import *
from testing.test_dataloaders import create_split_loaders
from testing.test_network import TestFeedforwardNet
from torch import tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class TestWeightAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        self.final = tensor([[0.8, 0.2],
                             [-0.1, 0.7]])
        self.t = tensor([[0.5, 0.5],
                         [0.1, 0.5]])
        self.t_1 = tensor([[0.4, 0.45],
                           [0.15, 0.4]])
        self.model = TestFeedforwardNet()
        self.model.load_state_dict(torch.load('../test_weights/MNIST_params'))
        transform = ToTensor()
        dataset = MNIST('.', download=True, transform=transform)

        batch_size = 1

        seed = 42
        p_val = 0.0
        p_test = 0.0
        extras = dict()

        train_loader, val_loader, test_loader = create_split_loaders(dataset=dataset,
                                                                     batch_size=batch_size,
                                                                     seed=seed,
                                                                     p_val=p_val, p_test=p_test,
                                                                     shuffle=False, extras=extras)
        batch = None
        for num, (images, labels) in enumerate(train_loader):
            if num == 1:
                break
            batch = images
        self.batch = torch.reshape(batch, (-1, 784))
        self.model.forward(self.batch)

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

    def test_analyse_decision(self):
        analyse_decision(self.model, self.batch, "band", "pos_neg", 3, "mongodb://localhost:27017/", "test_network1")
