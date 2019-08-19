import torch
from torch.nn.functional import *
import torch.nn as nn
import torch.nn.init as torch_init
from collections import OrderedDict


class TestFeedforwardNet(nn.Module):
    """
    A simple feed forward network for testing. For MNIST dataset.
    """
    def __init__(self):
        super(TestFeedforwardNet, self).__init__()
        self.activations = OrderedDict()
        # weights from input to hidden layer
        self.layer1 = nn.Linear(784, 100)
        # Use Xavier normal weights initialisation.
        torch_init.xavier_normal_(self.layer1.weight)

        # weights from hidden layer to outputs
        self.layer2 = nn.Linear(100, 100)
        # Use Xavier normal weights initialisation.
        torch_init.xavier_normal_(self.layer2.weight)

        self.layer3 = nn.Linear(100, 10)
        torch_init.xavier_normal_(self.layer2.weight)

    def forward(self, batch):
        """
        Forward pass through the network.
        Adds the activations to a dict for later relevance propagation.
        :param batch: Tensor Input batch
        :return: Tensor Output
        """
        batch = torch.tanh(self.layer1(batch))
        self.activations['layer1'] = batch
        batch = torch.tanh(self.layer2(batch))
        self.activations['layer2'] = batch
        batch = self.layer3(batch)
        self.activations['layer3'] = batch
        return batch


class TestDeepCNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison.

       Consists of three Conv2d layers, followed by one 4x4 max-pooling layer,
       and 2 fully-connected (FC) layers:

       conv1 -> conv2 -> conv3 -> maxpool -> conv4 -> conv5 -> fc1 -> fc2 (outputs)

       Make note:
       - Inputs are expected to be grayscale images (how many channels does this imply?)
       - The Conv2d layer uses a stride of 1 and 0 padding by default
       """

    def __init__(self):
        super(TestDeepCNN, self).__init__()

        self.activations = dict()

        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=8)

        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(12)

        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)

        # conv2: 12 input channels, 12 output channels, [8x8] kernel
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=8)
        self.conv2_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv2.weight)

        # conv3: 12 input channels, 10 output channels, [6x6] kernel
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=6)
        self.conv3_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv3.weight)

        # Apply max-pooling with a [2x2] kernel using tiling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4: 10 input channels, 10 output channels, [4x4] kernel
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4)
        self.conv4_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv4.weight)

        # conv5: 10 input channels, 8 output channels, [4x4] kernel
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=4)
        self.conv5_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv5.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define 2 fully connected layers:
        # TODO: Use the value you computed in Part 1, Question 4 for fc1's in_features
        self.fc1 = nn.Linear(in_features=115200, out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)

        # TODO: Output layer: what should out_features be?
        self.fc2 = nn.Linear(in_features=128, out_features=14)
        torch_init.xavier_normal_(self.fc2.weight)

    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """

        # Apply first convolution, followed by ReLU non-linearity;
        # use batch-normalization on its outputs
        # with torch.no_grad():
        batch = relu(self.conv1_normed(self.conv1(batch)))

        # Apply conv2 and conv3 similarly
        batch = relu(self.conv2_normed(self.conv2(batch)))

        batch = relu(self.conv3_normed(self.conv3(batch)))

        # Pass the output of conv3 to the pooling layer
        batch = self.pool(batch)

        batch = relu(self.conv4_normed(self.conv4(batch)))

        batch = relu(self.conv5_normed(self.conv5(batch)))

        batch = self.pool2(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))

        # Connect the reshaped features of the pooled conv3 to fc1
        batch = relu(self.fc1(batch))

        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc2(batch)

        # Return the class predictions
        return torch.sigmoid(batch)

    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features