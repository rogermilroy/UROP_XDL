import torch
from torch import Tensor
from heapq import heappush, heapreplace


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


def recurse_large_pos(tensor: Tensor, top: list, indices: list, n: int, dim: int) -> list:
    """
    Recursive function to find the largest n positive items in an n dimensional tensor, with the
    indices of their position within it.
    :param tensor: the Tensor within which we are searching.
    :param top: list A heap containing the largest items and their indices
    :param indices: list The current index to be examined.
    :param n: int The number of items to find.
    :param dim: int The current dimension of the original tensor.
    :return: list. A heap containing the largest items and their indices in the format (item,
    (indices))
    """
    # if its a number
    if len(tensor.size()) == 0:
        # heap already full.
        if len(top) >= n:
            # item is bigger than smallest item
            if tensor > top[0][0]:
                # add to heap
                heapreplace(top, (tensor, tuple(indices)))
                return top
            # make sure we return top in all cases.
            return top
        else:
            # add to heap
            heappush(top, (tensor, tuple(indices)))
            return top
    # not a single number (tensor of some shape)
    else:
        # iterate through the tensor
        for tens in tensor:
            # update t
            top = recurse_large_pos(tens, top, indices, n, dim + 1)
            # increment the relevant dimension
            indices[dim] += 1
        # reset index for next iteration.
        indices[dim] = 0
        return top


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
    Selects the largest n weights per layer as the most relevant ones.
    :param relevances:
    :param weights:
    :return:
    """
    for weight_layer in weights:
        pass


def top_relevant_neurons(relevances: list, weights: list, n: int) -> list:
    """
    Selects the weights between the top n most relevant neurons as the most relevant.
    :param relevances:
    :param weights:
    :return:
    """
    pass
