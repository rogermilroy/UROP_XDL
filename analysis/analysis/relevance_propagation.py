import torch
from torch import Tensor
import torch.nn.functional as func


def linear_relevance(activations: Tensor, weights: Tensor, relevance: Tensor, alpha, beta) -> \
        Tensor:
    """
    Calculate the relevances for a linear layer.
    :param activations: Tensor of activations for layer relevances are being propagated to.
    :param weights: Tensor of weights between the layers
    :param relevance: Tensor of relevances for layer above.
    :return: Tensor of relevances for layer.
    """

    act_size = activations.size()
    weight_size = weights.size()
    # check and fix most common matrix orientation issue
    if act_size[1] != weight_size[0] and act_size[1] == weight_size[1]:
        weights = weights.t()

    # take the positive weights TODO use epsilon?
    pos = torch.clamp(weights, min=0)
    pZ = torch.matmul(activations, pos) + 1e-9
    pS = relevance / pZ
    pC = torch.matmul(pS, pos.t())
    P = activations * pC * alpha
    # print("Positive relevance: ", P)

    neg = torch.clamp(weights, max=0)
    nZ = torch.matmul(activations, neg) + 1e-9
    nS = relevance / nZ
    nC = torch.matmul(nS, neg.t())
    N = activations * nC * beta
    # print("Negative relevance: ", N)

    return P - N


# at each layer we need to decide the shape/type of the layer.
def propagate_relevance(activations: Tensor, weights: Tensor, relevance: Tensor, alpha, beta) -> \
        Tensor:
    """
    Wrapper to decide which kind of relevance propagation is needed.
    :param activations: Tensor of activations.
    :param weights: Tensor of weights
    :return: Tensor of relevance values
    """
    # check linear layer TODO add flags?
    if len(activations.size()) == 2 and len(weights.size()) == 2:
        print("Linear layer with a batch")
        return linear_relevance(activations, weights, relevance, alpha, beta)
        # TODO how do we deal with batches?

    elif len(activations.size()) == 1 and len(weights.size()) == 2:
        print("Linear layer single entry.")
        return linear_relevance(activations, weights, relevance, alpha, beta)
    # TODO add convolutions and more.
    else:
        raise NotImplementedError("Only linear layers currently implemented.")


def layerwise_relevance(model, inputs) -> tuple:
    """
    Carries out the layer-wise relevance propagation procedure
    ref
    and returns the relevance's.
    :param model: The model (Neural Network)
    :param inputs: The inputs to the model.
    :return: dict The relevance values per layer as a dict of Tensors.
    """
    # TODO test for single decision only?
    # organise the data (preferably a list of alternating activations and weights)
    layers = list()
    weights = list()
    for param, tensor in reversed(model.state_dict().items()):
        if param.find("weight") != -1:
            layers.append(torch.tensor([0.]))
            layers.append(tensor)
            weights.append(tensor)
    for num, (layer, activation) in enumerate(reversed(model.activations.items())):
        layers[(num * 2)] = activation
    layers.append(inputs)

    # simple sanity check to make sure the layers are in the right order.
    for item in layers:
        print(item.size())

    # TODO think about dimensions for batch operations
    # TODO think about conv dimensions.

    relevances = list()
    layers[0] = func.softmax(layers[0])
    mask = torch.zeros_like(layers[0])
    mask[0, torch.argmax(layers[0])] = 1.
    rel = layers[0] * mask
    relevances.append(rel)
    # iterate through the list of layers and compute relevances.
    print(range(1, len(layers) - 1, 2))
    for i in range(1, len(layers) - 1, 2):
        rel = propagate_relevance(activations=layers[i+1], weights=layers[i], relevance=rel,
                                  alpha=1.0, beta=0.0)
        relevances.append(rel)

    return relevances, weights

