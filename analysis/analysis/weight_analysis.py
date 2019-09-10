# to contain all the weight analysis functions.
import torch
from torch import Tensor
from analysis.relevance_propagation import layerwise_relevance
from analysis.path_selection import *
from analysis.utils import *
import pymongo

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
def analyse_decision(model, inputs, selection: str, analysis: str, n: int, db_conn_string: str, collection:str):
    selection_func = {"band": band_selection,  "top_weights": top_weights,
                      "relevant_neurons": top_relevant_neurons}
    analysis_func = {"avg": avg_diff, "total": total_diff, "pos_neg": pos_neg_diff}

    # connect to mongodb db.
    db = pymongo.MongoClient(db_conn_string).training_data

    # compute relevances.
    weights, relevances = layerwise_relevance(model=model, inputs=inputs)

    # select weights (indices)
    weight_indices = selection_func[selection](weights, relevances, n)

    # pull data from db
    # TODO speed up the count_documents call currently arount 6s.
    items = db[collection].count_documents({'total_minibatch': {"$exists": True}})
    cursor = db[collection].find({"total_minibatch": {"$exists": True}},
                        {"total_minibatch", "model_state", "inputs"}).sort('total_minibatch', pymongo.DESCENDING)
    
    diffs = list()
    # iterate over all the minibatches.
    for i in range(items -1):
        if i == 2:
            # only do a few for testing.
            break
        # get the weights from the model state!
        weights1 = weights_from_model_state(cursor[i])
        weights2 = weights_from_model_state(cursor[i+1])
        # TODO maybe add the minibatch number in a tuple?
        # analyse the difference between this minibatch and the one before. i -> i+1
        diffs.append(analysis_func[analysis](current=weights1, past=weights2, final=weights))
    
    print(diffs)

    pass
