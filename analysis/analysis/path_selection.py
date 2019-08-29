from analysis.utils import *


def largest_n(tensor: Tensor, n: int, pos: bool) -> list:
    """
    Finds the n largest weights.
    :param tensor: Tensor n dimensional weights tensor
    :param n: int The number of weights to find.
    :param pos: bool Whether to use the positive or negative version.
    :return: list A heap of tuples containing the  indices of the largest weights.
    """
    indices = [0] * len(tensor.size())
    if pos:
        t = recurse_large_pos(tensor, list(), indices, n, 0)
    else:
        t = recurse_large_neg(tensor, list(), indices, n, 0)
    return [it[1] for it in t]


def band_search(relevances: list, weights: list, n: int) -> list:
    """
    Uses a similar idea to band search to identify the most relevant pathways through the network.
    Returns indices of weights per layer.
    :param relevances: list of Tensors. Relevance values for neurons.
    :param weights: list of Tensors. Weights.
    :param n: int TODO decide exactly what n will decide. Total paths? paths per layer?
    :return: List of indices.
    """
    pass


def top_weights(relevances: list, weights: list, n: int) -> list:
    """
    Selects the largest n weights per layer as the most relevant ones. Basically a benchmark for
    comparison.
    :param relevances: list A list of Tensors containing relevance values per neuron. Not Used.
    :param weights: list A list of Tensors containing the weights between the layers.
    :param n: int The number of weights desired per layer.
    :return: list A list containing per layer the indices of the largest weights.
    """
    result = list()
    for weight_layer in weights:
        result.append(largest_n(tensor=weight_layer, n=n, pos=True))
    return result


def top_relevant_neurons(relevances: list, weights: list, n: int) -> list:
    """
    Selects the weights between the top n most relevant neurons as the most relevant.
    :param relevances:
    :param weights:
    :param n: int
    :return:
    """
    # find the most relevant neurons per layer (positive)
    most_relev = list()
    for layer in relevances:
        most_relev.append(largest_n(layer, n, pos=True))
    print(most_relev)
    # this is where it gets complicated.
    # need to select the right indices regardless of dimensionality.
    pass
