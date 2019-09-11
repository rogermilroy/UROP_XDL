# Utils for analysis module.
import torch
from copy import deepcopy
from heapq import heappush, heapreplace

from heapq_max import heappush_max, heapreplace_max
from torch import Tensor
import re


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
                heapreplace(top, (tensor, deepcopy(indices)))
                return top
            # make sure we return top in all cases.
            return top
        else:
            # add to heap
            heappush(top, (tensor, deepcopy(indices)))
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


def recurse_large_neg(tensor: Tensor, top: list, indices: list, n: int, dim: int) -> list:
    """
    Recursive function to find the largest n negative items in an n dimensional tensor, with the
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
            # item is bigger than smallest item with negation to deal with negative items.
            if tensor < top[0][0]:
                # add to heap
                heapreplace_max(top, (tensor, deepcopy(indices)))
                return top
            # make sure we return top in all cases.
            return top
        else:
            # add to heap
            heappush_max(top, (tensor, deepcopy(indices)))
            return top
    # not a single number (tensor of some shape)
    else:
        # iterate through the tensor
        for tens in tensor:
            # update t
            top = recurse_large_neg(tens, top, indices, n, dim + 1)
            # increment the relevant dimension
            indices[dim] += 1
        # reset index for next iteration.
        indices[dim] = 0
        return top


def all_weights(l1: list, l2: list) -> list:
    """
    Method that assembles the indices of the weights connecting the neurons indexed in the two
    lists.
    :param l1: list Indices of neurons.
    :param l2: list Indices of neurons.
    :return: list Indices of weights.
    """
    # in simplest form ( two lists of lists containing a single number)
    # TODO deal with different dimensionality.
    result = list()
    if len(l1[0]) == 1:
        for index in l1:
            for idx in l2:
                result.append([index[0], idx[0]])
    else:
        raise NotImplementedError("Only linear layers currently supported.")
    return result


def path_to_weights(path: list) -> list:
    """
    Takes a list of neurons and returns the weights that connect them.
    TODO think about multidimensional inputs.
    :param path: list A list of neuron indices.
    :return: list A list of weight indices.
    """
    weights = list()
    for i in range(len(path) - 1):
        weights.append([path[i][0], path[i+1][0]])
    return weights


def find_index(tensor: Tensor, index: list) -> Tensor:
    """
    This function is annoying but necessary because its not built into torch or numpy or anywhere.
    :param tensor: Tensor The tensor containing the item.
    :param index: list The index of the item.
    :return: Tensor The item.
    """
    dims = len(index)
    if dims == 1:
        return tensor[index[0]]
    elif dims == 2:
        return tensor[index[0], index[1]]
    elif dims == 3:
        return tensor[index[0], index[1], index[2]]
    elif dims == 4:
        return tensor[index[0], index[1], index[2], index[3]]


def find_indices(tensor: Tensor, indices: list) -> list:
    """
    Finds a list of indices in a given Tensor.
    :param tensor: Tensor The tensor containing the items.
    :param indices: list The indices of the items
    :return: list A list of the items
    """
    w = list()
    for index in indices:
        w.append(find_index(tensor=tensor, index=index))
    return w


def get_indices_from_weights(weights: list, indices: list):
    # Assumption is that weights are a list of Tensors in the correct order.
    # And that indices is a 3 dimensional list
    w = list()
    # print("Weights: ",weights)
    # print(len(weights))
    for i in range(len(weights)):
        print("Tensor: ", weights[i].size())
        print("Indices: ", indices[i])
        w.append(find_indices(weights[i], indices[i]))
    return torch.tensor(w)


def weights_from_model_state(model_state: dict) -> list:
    # extracts the weights from the model state
    weights = [None] * len(model_state)
    for param, tensor in model_state.items():
        # print(param)
        # find returns -1 if not present
        if param.find("weight") != -1:
            # look for the number as usually something like 'layer1.weight'
            nums = re.findall(r'\d+', param)
            if len(nums) != 1:
                raise Exception("Some issue with parameter numbering.")
            else:
                weights.insert(int(nums[0]) - 1, tensor)
    
    weights = [x for x in reversed(weights) if x is not None]
    # print(weights)
    return weights