import unittest

import matplotlib.pyplot as plt
import torch
from analysis.relevance_propagation import *
from testing.test_dataloaders import create_split_loaders
from testing.test_network import TestFeedforwardNet
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def visualise_mnist_relevance(relevances:list):
    relevance = relevances[-1].detach()
    relevance = torch.reshape(relevance, (28, 28))
    # print(relevance.size())
    plt.imshow(relevance)
    plt.show()


class TestRelevancePropagation(unittest.TestCase):

    def setUp(self) -> None:
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

    def test_param_processing(self):
        relevances, weights = layerwise_relevance(model=self.model, inputs=self.batch)
        # print("Relevances: ", len(relevances))
        # print("Weights: ", len(weights))
        visualise_mnist_relevance(relevances)

    def test_linear_relevance(self):
        activations = torch.tensor([[0.7, 0.1, 0.3, 0.8]])
        weights = torch.tensor([[0., 1., 0., 0.],
                                [1., -1., -2., 1.],
                                [0., -2., 1., 0]]).t()
        relevances = torch.tensor([[0., 0.8, 0.]])
        self.assertTrue(-0.001 < float(torch.sum(linear_relevance(activations, weights,
                                                                  relevances, 1.0, 0.0) -
                         torch.tensor([[0.3733, 0.0000, 0.0000, 0.4267]]))) < 0.001)

    def test_batch_linear_relevance(self):
        activations = torch.tensor([[0.7, 0.1, 0.3, 0.8],
                                    [0.2, 0.7, 0.1, 0.3]])
        weights = torch.tensor([[0., 1., 0., 0.],
                                [1., -1., -2., 1.],
                                [0., -2., 1., 0]]).t()
        relevances = torch.tensor([[0., 0.8, 0.]])
        self.assertTrue(-0.001 < float(torch.sum(linear_relevance(activations, weights,
                                                                  relevances, 1.0, 0.0)[1] -
                         torch.tensor([[0.3200, 0.0000, 0.0000, 0.4800]]))) < 0.001)
