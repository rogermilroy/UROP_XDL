# to contain all the weight analysis functions.
import pymongo
from analysis.path_selection import *
from analysis.relevance_propagation import new_layerwise_relevance, layerwise_relevance
from testing.test_network import TestFeedforwardNet
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from utils.data_processing import decode_model_state, decode_tensor


# creating indices? Maybe in datalake code?

# mongo manipulation (maybe dropping epochs after final epoch)


# distance from final network
from visualisation.plot import Plotter


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
    selection_func = {"band": band_selection, "top_weights": top_weights,
                      "relevant_neurons": top_relevant_neurons}
    analysis_func = {"avg": avg_diff, "total": total_diff, "pos_neg": pos_neg_diff}

    # connect to mongodb db.
    db = pymongo.MongoClient(db_conn_string).training_data

    model_state = decode_model_state(
        db[collection].find_one({'final_model_state': {'$exists': True}})[
            'final_model_state'])

    model.load_state_dict(model_state)

    # compute relevances.
    relevances, weights = layerwise_relevance(model=model, inputs=inputs)  # TODO change
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
        model2 = decode_model_state(cursor[i + 1]['model_state'])
        weights1 = extract_weights(weights_from_model_state(model1), weight_indices)
        weights2 = extract_weights(weights_from_model_state(model2), weight_indices)
        # print("Weights1.size: ",weights1.size())
        # print("Weights2.size: ",weights2.size())
        # print("Final weights.size: ",final_weights.size())

        # analyse the difference between this minibatch and the one before. i -> i+1
        # TODO encode tensor for storage?? or do later?
        diff = analysis_func[analysis](current=weights1, past=weights2,
                                       final=final_weights)
        diffs.append((cursor[i]['total_minibatch'],
                      decode_tensor(cursor[i]['inputs']),
                      decode_tensor(cursor[i]['outputs']),
                      decode_tensor(cursor[i]['targets']),
                      diff))

    # print(diffs)

    return diffs, relevances


def visualise_differences(model, inputs, db, collection):
    """
    Visualise training.
    :param model: The model that was trained. With final state weights loaded. IMPORTANT
    :param inputs: The original inputs
    :param db: The data base containing the training records.
    :param collection: The collection containing the training records.
    :return:
    """

    outs = torch.softmax(model.forward(inputs), dim=1)

    # compute relevances.
    relevances = list()
    for i in range(3):
        temp, _ = new_layerwise_relevance(model=model, inputs=inputs, index=i)
        relevances.append(temp)

    top3 = torch.topk(outs, k=3).indices

    # pull data from db
    items = db[collection].count_documents({'total_minibatch': {"$exists": True}},
                                           hint='total_minibatch_-1')
    # print(items)
    cursor = db[collection].find({"total_minibatch": {"$exists": True}},
                                 {"total_minibatch", "model_state", "inputs", "outputs",
                                  "targets"}).sort('total_minibatch', pymongo.ASCENDING)

    diffs = dict()
    preds = list()
    targets = list()
    tmp = torch.zeros_like(outs)  # TODO check
    src = torch.ones_like(outs)

    # iterate over items
    for i in tqdm(range(items - 1)):
        if i == 1000:
            # only do a few for testing.
            break

        # check if the target and outputs match. flag if they don't.
        curr_in = decode_tensor(cursor[i]['inputs']).to(torch.float)  # to stop double float
        # issue.
        model.load_state_dict(decode_model_state(cursor[i]['model_state']))
        preds.append(decode_tensor(cursor[i]['outputs']))
        targets.append(
            (cursor[i]['total_minibatch'], decode_tensor(cursor[i]['targets']).detach().item()))

        # carry out relevance on error and post error model.
        relevance0 = list()
        relevance1 = list()
        for j in range(3):
            temp, _ = new_layerwise_relevance(model=model, inputs=curr_in, index=j)
            relevance0.append(temp)
        # load post error model state
        model.load_state_dict(decode_model_state(cursor[i + 1]['model_state']))
        after = model.forward(curr_in)
        for j in range(3):
            temp, _ = new_layerwise_relevance(model=model, inputs=curr_in, index=j)
            relevance1.append(temp)
        diffs[cursor[i]['total_minibatch']] = (relevance0,
                                               relevance1,
                                               torch.topk(torch.softmax(
                                                   decode_tensor(cursor[i]['outputs']),
                                                   dim=1), k=3).indices,
                                               torch.topk(torch.softmax(after, dim=1), k=3).indices,
                                               decode_tensor(cursor[i]['outputs']),
                                               torch.softmax(after, dim=1),
                                               decode_tensor(cursor[i]['targets']))

    return diffs, top3, relevances, torch.cat(preds, dim=0).T, list(sorted(targets))


if __name__ == '__main__':
    model = TestFeedforwardNet()  #TODO change here

    db = pymongo.MongoClient("mongodb://localhost:27017/").training_data

    model_state = decode_model_state( # TODO change
        db["corrupted_dataset0"].find_one({'final_model_state': {'$exists': True}})[
            'final_model_state'])

    model.load_state_dict(model_state)

    dataset = MNIST('../../MNIST/original', download=False, transform=ToTensor())

    sample, lab = dataset[25]
    print(torch.argmax(torch.softmax(model.forward(sample.reshape(-1, 784)), 1), dim=1)) #TODO
    # change
    print(lab)
    diffs, relevances = analyse_decision(model, sample.reshape(-1, 784), "band", #TODO change here
                                         "total", 2, "mongodb://localhost:27017/",
                                         "corrupted_dataset0")  #TODO changn


    stuff = dict()
    pos_neg = list()

    for minibatch, inputs, outputs, targets, diff in diffs:

        stuff[minibatch] = (inputs.reshape(28, -1), torch.argmax(outputs), targets[0])

        pos_neg.append(diff)

    plotter = Plotter(pos_neg,
                      vis_data=stuff,
                      relevances=relevances[-1].detach().numpy().reshape(28, -1),
                      pos_neg=False)
    plotter.plot()

