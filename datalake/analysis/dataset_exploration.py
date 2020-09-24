import matplotlib.pyplot as plt
import pymongo
import torch
from testing.test_network import TestDeepCNN
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils.data_processing import decode_model_state


def explore_dataset(model, dataset):

    for i, (sample, lab) in enumerate(dataset):
        plt.imshow(sample.squeeze().numpy())
        plt.show()
        print(torch.argmax(torch.softmax(model.forward(sample.unsqueeze(0)), dim=1), dim=1))
        print(lab)
        print(i)
        input("Continue?\n")


if __name__ == '__main__':

    model = TestDeepCNN()  # TODO change here

    db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    model_state = decode_model_state(  # TODO change
        db["conv_full0_final"].find_one({'final_model_state': {'$exists': True}})[
            'final_model_state'])

    model.load_state_dict(model_state)

    dataset = MNIST('../../MNIST/original', download=False, transform=ToTensor())

    explore_dataset(model=model, dataset=dataset)
