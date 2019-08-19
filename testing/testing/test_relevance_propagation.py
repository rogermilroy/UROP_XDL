import unittest
from testing.test_network import TestFeedforwardNet
from analysis.relevance_propagation import layerwise_relevance
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from testing.test_dataloaders import create_split_loaders
import torch


class TestRelevancePropagation(unittest.TestCase):

    def setUp(self) -> None:
        self.model = TestFeedforwardNet()
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
        batch = torch.reshape(batch, (-1, 784))
        self.model.forward(batch)

    def tearDown(self) -> None:
        pass

    def test_param_processing(self):
        layerwise_relevance(model=self.model)

    def test_linear_relevance(self):
