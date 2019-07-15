import torch
from torch.nn.functional import *
import torch.nn as nn
import torch.nn.init as torch_init


class TestFeedforwardNet(nn.Module):
    """
    A simple feed forward network for testing. For MNIST dataset.
    """
    def __init__(self):
        super(TestFeedforwardNet, self).__init__()
        # weights from input to hidden layer
        self.layer1 = nn.Linear(784, 100)
        # Use Xavier normal weights initialisation.
        torch_init.xavier_normal_(self.layer1.weight)

        # weights from hidden layer to outputs
        self.layer2 = nn.Linear(100, 10)
        # Use Xavier normal weights initialisation.
        torch_init.xavier_normal_(self.layer2.weight)

    def forward(self, batch):
        batch = torch.sigmoid(self.layer1(batch))
        batch = torch.sigmoid(self.layer2(batch))
        return batch

