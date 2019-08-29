# Utils for analysis module.

from copy import deepcopy
from heapq import heappush, heapreplace

from heapq_max import heappush_max, heapreplace_max
from torch import Tensor


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

