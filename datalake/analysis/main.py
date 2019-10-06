# Interface for scheduling analysis.
from testing.test_network import TestFeedforwardNet
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from testing.test_dataloaders import create_split_loaders
import zmq
from utils.network_details import get_ip
from utils.data_processing import decode_model_state
import ujson as json
from analysis.weight_analysis import analyse_decision
# import argparse
import time
import pymongo


def temporary_analysis(selection: str, analysis: str, n: int, training_run: str):
    """
    A quick script to get some analyses done
    :return:
    """

    context = zmq.Context()
    publish = context.socket(zmq.PUSH)
    publish.bind('tcp://' + str(get_ip()) + ':' + '5556')
    print(str(get_ip()) + ':' + '5556')

    db = pymongo.MongoClient("mongodb://localhost:27017/").training_data
    model_state = decode_model_state(db[training_run].find_one({'final_model_state': {'$exists': True}},
            projection={'final_model_state': True, '_id': False})['final_model_state'])

    model = TestFeedforwardNet()
    model.load_state_dict(model_state)
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

    # run batch through model.
    model.forward(batch)

    # get results of analyse_decision
    res = analyse_decision(model, batch, selection, analysis, n,
                           "mongodb://localhost:27017/",
                           training_run)
    packet = {'training_run': training_run, 'weight_selection_type': selection,
              'analysis_type': analysis, 'data': res}
    print(packet)

    # Put in push socket.
    publish.send_string(json.dumps(packet))

    time.sleep(5.)


if __name__ == '__main__':

    temporary_analysis("band", "pos_neg", 3, "test_network1")
    temporary_analysis("band", "avg", 3, "test_network1")
    temporary_analysis("relevant_neurons", 'pos_neg', 5, 'test_network1')
    temporary_analysis("band", "total", 6, "test_network1")
    temporary_analysis("top_weights", "avg", 5, "test_network1")
    temporary_analysis("relevant_neurons", 'total', 5, 'test_network1')
