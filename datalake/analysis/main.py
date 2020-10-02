# Interface for scheduling analysis.
# import argparse
import time

import gridfs
import pymongo
import torch
import ujson as json
import zmq
from analysis.weight_analysis import analyse_decision, visualise_differences, top_3_relevances
from testing.test_dataloaders import create_split_loaders
from testing.test_network import TestFeedforwardNet, TestDeepCNN
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils.data_processing import decode_model_state, encode_tensor, encode_diffs, \
    decode_diffs, \
    decode_tensor
from utils.network_details import get_ip
from visualisation.plot import CNNPlotter


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
    dataset = MNIST('../../MNIST/original', download=False, transform=transform)

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


def analyse_and_store(model, db: pymongo.MongoClient, source_collection: str,
                      sample: torch.Tensor, label: torch.Tensor,
                      outs: torch.Tensor) -> None:
    """
    Method to run an analysis and store in a local MongoDB.
    NOTE important side effect is that it changes the weights in the model
    :param model: The model with the final state weights loaded
    :param db: The MongoDB client instance
    :param source_collection: The collection with the model states from training.
    :param sink_collection: The collection for storing the analysis.
    :param sample: The inputs
    :param label: The true label for the inputs.
    :param outs: The output of the model (softmax over outs)
    :return: None
    """

    diffs, preds, targets = visualise_differences(model=model, db=db, collection=source_collection)

    analysis = {"training_preds": encode_tensor(preds),
                "training_targets": targets,
                "vis_data": encode_diffs(diffs),  # dict of tensors ish
                "preds": encode_tensor(outs),
                "target": encode_tensor(label)}

    analysis_gridfs = gridfs.GridFS(database=db, collection=source_collection+'_analyses')

    analysis_gridfs.put(json.dumps(analysis), encoding='utf-8', analysis_type='visualise_diffs')


def visualise_analysis(model, sample, db, collection: str, analysis_type: str):

    relevances, top3 = top_3_relevances(model, sample)

    analysis_gridfs = gridfs.GridFS(database=db, collection=collection)

    b = analysis_gridfs.find_one({'analysis_type': analysis_type})
    b = json.loads(str(b.read(), encoding='utf-8'))

    plotter = CNNPlotter(training_preds=decode_tensor(b['training_preds']).numpy(),
                         training_targets=b['training_targets'],
                         vis_data=decode_diffs(b['vis_data']),
                         top3=top3,
                         relevances=relevances,
                         preds=decode_tensor(b['preds']),
                         target=decode_tensor(b['target']),
                         color_map='coolwarm')
    plotter.plot()


if __name__ == '__main__':

    model = TestDeepCNN()

    db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    model_state = decode_model_state(
        db["conv_full0_final"].find_one({'final_model_state': {'$exists': True}})['final_model_state'])

    model.load_state_dict(model_state)

    dataset = MNIST('../../MNIST/original', download=False, transform=ToTensor())

    sample, lab = dataset[24]

    sample = sample.unsqueeze(0)

    visualise_analysis(model=model, sample=sample, db=db, collection='conv_full0_analyses',
                       analysis_type="visualise_diffs")

    # outs = torch.softmax(model.forward(sample), dim=1)
    #
    # analyse_and_store(model=model, db=db, source_collection='conv_full0',
    #                   sample=sample, label=lab, outs=outs)

    # temporary_analysis("band", "pos_neg", 3, "test_network1")
    # temporary_analysis("band", "avg", 3, "test_network1")
    # temporary_analysis("relevant_neurons", 'pos_neg', 5, 'test_network1')
    # temporary_analysis("band", "total", 6, "test_network1")
    # temporary_analysis("top_weights", "avg", 5, "test_network1")
    # temporary_analysis("relevant_neurons", 'total', 5, 'test_network1')
