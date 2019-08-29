from analysis.utils import *


def top_n_weights(weights: Tensor, n: int) -> list:
    """
    Finds the n largest weights.
    :param weights: Tensor n dimensional weights tensor
    :param n: int The number of weights to find.
    :return: list A heap containing the largest weights and their indices in the format (weight,
    (indices))
    """
    dims = len(weights.size())
    indices = [0] * dims
    return recurse_large_pos(weights, list(), indices, n, 0)


def band_search(relevances: list, weights: list, n: int) -> list:
    """
    Uses a similar idea to band search to identify the most relevant pathways through the network.
    Returns indices of weights per layer.
    :param relevances: list of Tensors. Relevance values for neurons.
    :param weights: list of Tensors. Weights.
    :return: List of indices.
    """
    pass


def top_weights(relevances: list, weights: list, n: int) -> list:
    """
    Selects the largest n weights per layer as the most relevant ones. Basically a benchmark for
    comparison.
    :param relevances: list A list of Tensors containing relevance values per neuron. Not Used.
    :param weights: list A list of Tensors containing the weights between the layers.
    :return: list A list containing per layer the indices of the largest weights.
    """
    result = list()
    for weight_layer in weights:
        t = top_n_weights(weights=weight_layer, n=n)
        res = [it[1] for it in t]
        result.append(res)
    return result


def top_relevant_neurons(relevances: list, weights: list, n: int) -> list:
    """
    Selects the weights between the top n most relevant neurons as the most relevant.
    :param relevances:
    :param weights:
    :return:
    """
    pass