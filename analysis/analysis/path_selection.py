import math
import torch

from analysis.utils import *


def largest_n(tensor: Tensor, n: int, pos: bool, in_order: bool = False, desc: bool = True) -> list:
    """
    Finds the n largest weights.
    :param tensor: Tensor n dimensional weights tensor
    :param n: int The number of weights to find.
    :param pos: bool Whether to use the positive or negative version.
    :param in_order: bool Whether to sort the output.
    :param desc: bool Whether the sorted output should be descending order.
    :return: list A heap of tuples containing the  indices of the largest weights.
    """
    indices = [0] * len(tensor.size())
    if pos:
        t = recurse_large_pos(tensor, list(), indices, n, 0)
    else:
        t = recurse_large_neg(tensor, list(), indices, n, 0)
    if in_order:
        t.sort(reverse=desc)
    return [it[1] for it in t]


def band_selection(relevances: list, weights: list, n: int) -> list:
    """
    Uses a similar idea to band search to identify the most relevant pathways through the network.
    Returns indices of weights per layer.
    :param relevances: list of Tensors. Relevance values for neurons.
    :param weights: list of Tensors. Weights.
    :param n: int The branching factor or band width.
    :return: List of indices.
    """
    # select largest n*layer neurons per layer. using largest n
    layer_largest = list()
    for i in range(len(relevances)):
        if i == 0:
            layer_largest.append(largest_n(relevances[i], n=1, pos=True))
        else:
            layer_largest.append(sorted(largest_n(relevances[i], n=i*n, pos=True, in_order=True,
                                                  desc=True)))

    # have a current path variable
    current_path = list()
    paths = list()
    depth = len(layer_largest)

    indices = [0] * depth

    # iterate over the total number of paths.
    for i in range(len(layer_largest[-1])):
        # permute the indices.
        for z in range(len(layer_largest)):
            power = (depth - 1) - z
            indices[z] = int(i // math.pow(n, power))
        # iterate over the layers.
        for j in range(len(layer_largest)):
            # add the item at indices in that layer to the path.
            current_path.append(layer_largest[j][indices[j]])
        # add the weights of the path to the list and clear it for the next path.
        paths.append(path_to_weights(current_path))
        current_path = list()

    # TODO Need to check for narrower sections and potentially saturating the neurons
    # make n the maximum branching factor.

    return paths


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
    :param relevances: list A list of Tensors containing relevance values per neuron. Not Used.
    :param weights: list A list of Tensors containing the weights between the layers.
    :param n: int The number of neurons to take per layer.
    :return: list A list of indices of weights in the path to track.
    """
    # TODO have a different number of relevant neurons for the output layer. Probably only want one?
    # find the most relevant neurons per layer (positive)
    most_relev = list()
    for layer in relevances:
        most_relev.append(largest_n(layer, n, pos=True))
    # assemble the indices of weights that connect the neurons.
    result = list()
    for i in range(len(most_relev) - 1):
        result.append(all_weights(most_relev[i], most_relev[i+1]))
    return result


def extract_weights(weights_list: list, indices_list: list) -> Tensor:
    """
    Takes the indices of the weights desired and assembles them into a Tensor for further
    processing.
    :param weights_list: list A list of Tensors containing the weights between layers.
    :param indices_list: list A list of lists containing the indices of the weights we are
    interested in.
    :return: Tensor A Tensor of weights.
    """
    w = list()
    layers = len(weights_list)
    for i in range(layers):
        layer = weights_list[i]
        layer_indices = indices_list[i]
        w.append(find_indices(tensor=layer, indices=layer_indices))
    return torch.tensor(w).flatten()
