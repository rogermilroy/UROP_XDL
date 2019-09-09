# to contain all the weight analysis functions.
import torch
from torch import Tensor
from analysis.relevance_propagation import layerwise_relevance
from analysis.path_selection import *
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
def analyse_decision(model, inputs, selection: str, n: int, db_conn_string: str, collection:str):
    selection_func = {"band": band_selection,  "top_weights": top_weights,
                      "relevant_neurons": top_relevant_neurons}

    # connect to mongodb db.
    db = pymongo.MongoClient(db_conn_string).training_data

    # compute relevances.
    weights, relevances = layerwise_relevance(model=model, inputs=inputs)

    # select weights (indices)
    weight_indices = selection_func[selection](weights, relevances, n)

    # pull data from db
    db[collection].find({"total_minibatch": {"$exists": True}},
                        {"total_minibatch", "model_state"}).sort({"total_minibatch": -1})

    pass
