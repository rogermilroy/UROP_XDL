# to contain all the weight analysis functions.
import matplotlib.pyplot as plt
import pymongo
import torchvision
from analysis.path_selection import *
from analysis.relevance_propagation import new_layerwise_relevance, layerwise_relevance
from testing.test_network import TestDeepCNN, TestFeedforwardNet
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from utils.data_processing import decode_model_state, decode_tensor
from visualisation.plot import Plotter, CNNPlotter


# creating indices? Maybe in datalake code?

# mongo manipulation (maybe dropping epochs after final epoch)


# distance from final network
def dist_to_final(current: Tensor, final: Tensor) -> Tensor:
    """
    Computes the distance the current weights are from the final models weights for selected
    weights.
    :param current: Tensor The current selected weights
    :param final: Tensor The final models weights
    :return: Tensor The absolute distance per weight.
    """
    return torch.abs(final - current)


# comparison of two minibatches (progress)
def step_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    """
    Computes the difference between two instances weights compared to the final models weights.
    :param current: Tensor The first (most recent) selection of weights.
    :param past: Tensor The previous (least recent) selection of weights.
    :param final: Tensor The final models selection of weights.
    :return: Tensor The difference between the selections.
    """
    current = current.to(torch.float)
    past = past.to(torch.float)
    final = final.to(torch.float)
    # TODO find a better place for this.
    return dist_to_final(current, final) - dist_to_final(past, final)


# metrics over the selected weights
def avg_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    """
    Computes the average difference between two instances weights compared to the final models
    weights.
    :param current: Tensor The first (most recent) selection of weights.
    :param past: Tensor The previous (least recent) selection of weights.
    :param final: Tensor The final models selection of weights.
    :return: Tensor The average difference.
    """
    return torch.mean(step_diff(current, past, final))


def total_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    """
    Computes the total difference between two instances weights compared to the final models
    weights.
    :param current: Tensor The first (most recent) selection of weights.
    :param past: Tensor The previous (least recent) selection of weights.
    :param final: Tensor The final models selection of weights.
    :return: Tensor The total difference.
    """
    return torch.sum(step_diff(current, past, final))


def pos_neg_diff(current: Tensor, past: Tensor, final: Tensor) -> Tensor:
    """
    Computes the total positive and negative differences between two instances weights compared to
    the final models weights.
    :param current: Tensor The first (most recent) selection of weights.
    :param past: Tensor The previous (least recent) selection of weights.
    :param final: Tensor The final models selection of weights.
    :return: Tensor The positive followed by the negative difference.
    """
    diff = step_diff(current=current, past=past, final=final)
    pos = torch.clamp_min(diff, min=0)
    neg = torch.clamp_max(diff, max=0)
    return torch.tensor([torch.sum(pos), torch.sum(neg)])


# analysis of a large number (maybe all minibatches)
def analyse_decision(model, inputs, selection: str, analysis: str, n: int, db_conn_string: str,
                     collection: str):
    selection_func = {"band": band_selection,  "top_weights": top_weights,
                      "relevant_neurons": top_relevant_neurons}
    analysis_func = {"avg": avg_diff, "total": total_diff, "pos_neg": pos_neg_diff}

    # connect to mongodb db.
    db = pymongo.MongoClient(db_conn_string).training_data

    model_state = decode_model_state(db[collection].find_one({'final_model_state': {'$exists': True}})[
        'final_model_state'])

    model.load_state_dict(model_state)

    # compute relevances.
    relevances, weights = layerwise_relevance(model=model, inputs=inputs)  #TODO change
    # here......

    # select weights (indices)
    weight_indices = selection_func[selection](relevances=relevances, weights=weights, n=n)

    final_weights = extract_weights(weights, weight_indices)
    # pull data from db
    items = db[collection].count_documents({'total_minibatch': {"$exists": True}},
                                           hint='total_minibatch_-1')
    # print(items)
    cursor = db[collection].find({"total_minibatch": {"$exists": True}},
                                 {"total_minibatch", "model_state", "inputs", "outputs",
                                  "targets"}).sort('total_minibatch', pymongo.DESCENDING)
    
    diffs = list()

    # iterate over all the minibatches.
    for i in range(items - 1):
        # if i == 1:
        #     # only do a few for testing.
        #     break

        model1 = decode_model_state(cursor[i]['model_state'])
        model2 = decode_model_state(cursor[i+1]['model_state'])
        weights1 = extract_weights(weights_from_model_state(model1), weight_indices)
        weights2 = extract_weights(weights_from_model_state(model2), weight_indices)
        # print("Weights1.size: ",weights1.size())
        # print("Weights2.size: ",weights2.size())
        # print("Final weights.size: ",final_weights.size())

        # analyse the difference between this minibatch and the one before. i -> i+1
        # TODO encode tensor for storage?? or do later?
        diff = analysis_func[analysis](current=weights1,past=weights2,
                                                     final=final_weights)
        diffs.append((cursor[i]['total_minibatch'],
                      decode_tensor(cursor[i]['inputs']),
                      decode_tensor(cursor[i]['outputs']),
                      decode_tensor(cursor[i]['targets']),
                      diff))

    # print(diffs)

    return diffs, relevances


def visualise_differences(model, inputs, db_conn_string, collection):
    # first connect to the db

    # connect to mongodb db.
    db = pymongo.MongoClient(db_conn_string).training_data

    model_state = decode_model_state(
        db[collection].find_one({'final_model_state': {'$exists': True}})[
            'final_model_state'])

    model.load_state_dict(model_state)

    # compute relevances.
    relevances = list()
    for i in range(3):
        temp, _, outs = new_layerwise_relevance(model=model, inputs=inputs, index=i)
        relevances.append(temp)

    top3 = torch.topk(outs, k=3).indices

    # pull data from db
    items = db[collection].count_documents({'total_minibatch': {"$exists": True}},
                                           hint='total_minibatch_-1')
    # print(items)
    cursor = db[collection].find({"total_minibatch": {"$exists": True}},
                                 {"total_minibatch", "model_state", "inputs", "outputs",
                                  "targets"}).sort('total_minibatch', pymongo.DESCENDING)

    diffs= dict()

    # iterate over items
    for i in tqdm(range(items - 1)):
        # if i == 100:
        #     # only do a few for testing.
        #     break

        # check if the target and outputs match. flag if they don't.
        curr_in = decode_tensor(cursor[i]['inputs']).to(torch.float) # to stop double float
        # issue.
        model.load_state_dict(decode_model_state(cursor[i]['model_state']))
        # carry out relevance on error and post error model.
        relevance0 = list()
        relevance1 = list()
        for j in range(3):
            temp, _, _ = new_layerwise_relevance(model=model, inputs=curr_in, index=j)
            relevance0.append(temp)
        # load post error model state
        model.load_state_dict(decode_model_state(cursor[i+1]['model_state']))
        for j in range(3):
            temp, _, _= new_layerwise_relevance(model=model, inputs=curr_in, index=j)
            relevance1.append(temp)
        diffs[cursor[i]['total_minibatch']] = (1.0 if torch.argmax(decode_tensor(cursor[i]['outputs'])) != torch.argmax(decode_tensor(
            cursor[i]['targets'])) else 0.0,
                                               relevance0,
                                               relevance1,
                                               torch.topk(decode_tensor(cursor[i]['outputs']), 3).indices,
                                               decode_tensor(cursor[i]['outputs']),
                                               decode_tensor(cursor[i]['targets']))

    return diffs, top3, relevances


if __name__ == '__main__':
    # model = TestFeedforwardNet()  #TODO change here
    #
    # db = pymongo.MongoClient("mongodb://localhost:27017/").training_data
    #
    # model_state = decode_model_state( # TODO change
    #     db["corrupted_dataset0"].find_one({'final_model_state': {'$exists': True}})[
    #         'final_model_state'])
    #
    # model.load_state_dict(model_state)
    #
    # dataset = MNIST('../../MNIST/original', download=False, transform=ToTensor())
    #
    # # for i, (sample, lab) in enumerate(dataset):
    # #     plt.imshow(sample.squeeze().numpy())
    # #     plt.show()
    # #     print(torch.argmax(torch.softmax(model.forward(sample.reshape(-1, 784)), 1), dim=1))
    # #     print(lab)
    # #     print(i)
    # #     input("Continue?\n")
    #
    # sample, lab = dataset[25]
    # # plt.imshow(sample.squeeze().numpy())
    # # plt.show()
    # print(torch.argmax(torch.softmax(model.forward(sample.reshape(-1, 784)), 1), dim=1)) #TODO
    # # change
    # print(lab)
    # diffs, relevances = analyse_decision(model, sample.reshape(-1, 784), "band", #TODO change here
    #                                      "total", 2, "mongodb://localhost:27017/",
    #                                      "corrupted_dataset0")  #TODO changn
    #
    # # plt.imshow(relevances[-1].detach().numpy().reshape(28, -1), cmap='coolwarm', alpha=0.8)
    # # plt.show()
    #
    # stuff = dict()
    # pos_neg = list()
    #
    # for minibatch, inputs, outputs, targets, diff in diffs:
    #
    #     stuff[minibatch] = (inputs.reshape(28, -1), torch.argmax(outputs), targets[0])
    #
    #     pos_neg.append(diff)
    #
    #     # plt.imshow(inputs.reshape(28, -1))
    #     # plt.show()
    #     # print("Model: ", torch.argmax(outputs))
    #     # print("Target: ", targets)
    #     # print("Diff: ", diff)
    #     # input("Next")
    #
    # plotter = Plotter(pos_neg,
    #                   vis_data=stuff,
    #                   relevances=relevances[-1].detach().numpy().reshape(28, -1),
    #                   pos_neg=False)
    # plotter.plot()




    model = TestDeepCNN()  # TODO change here

    db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    model_state = decode_model_state(  # TODO change
        db["conv_test0"].find_one({'final_model_state': {'$exists': True}})[
            'final_model_state'])

    model.load_state_dict(model_state)

    dataset = MNIST('../../MNIST/original', download=False, transform=ToTensor())

    # for i, (sample, lab) in enumerate(dataset):
    #     plt.imshow(sample.squeeze().numpy())
    #     plt.show()
    #     print(torch.argmax(torch.softmax(model.forward(sample.reshape(-1, 784)), 1), dim=1))
    #     print(lab)
    #     print(i)
    #     input("Continue?\n")

    sample, lab = dataset[0]
    # plt.imshow(sample.squeeze().numpy())
    # plt.show()
    print(torch.softmax(model.forward(sample.unsqueeze(0)), dim=1))  # TODO change
    print(lab)
    diffs, top3, relevances = visualise_differences(model, sample.unsqueeze(0),
                                               "mongodb://localhost:27017/",
                                         "conv_test0")

    # for minibatch, stuff in diffs.items():
    #     print(minibatch, stuff[0])

    # plt.imshow(relevances[0][-1].detach().numpy().reshape(28, -1), cmap='coolwarm', alpha=0.8)
    # plt.show()

    flags = list()

    for minibatch, (flag, _, _, _, _, _) in diffs.items():
        flags.append((minibatch, flag))

    flags = list(zip(*sorted(flags)))

    # print(flags)

    plotter = CNNPlotter(list(flags[1]),
                         vis_data=diffs,
                         top3=top3,
                         relevances=relevances,
                         preds=torch.softmax(model.forward(sample.unsqueeze(0)), dim=1),
                         target=lab,
                         pos_neg=False)
    plotter.plot()

