import torch
from torch import Tensor

# iterate through weights and activations from final outputs to inputs


def linear_relevance(activations: Tensor, weights: Tensor, relevance: Tensor) -> Tensor:
    """
    Calculate the relevances for a linear layer.
    :param activations: Tensor of activations for layer relevances are being propagated to.
    :param weights: Tensor of weights between the layers
    :param relevance: Tensor of relevances for layer above.
    :return: Tensor of relevances for layer.
    """
    # TODO have changeable alpha and beta
    alpha = 1.0
    beta = 0.0

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
def propagate_relevance(activations: Tensor, weights: Tensor, relevance: Tensor):
    """
    Wrapper to decide which kind of relevance propagation is needed.
    :param activations: Tensor of activations.
    :param weights: Tensor of weights
    :return: Tensor of relevance values
    """
    # check linear layer TODO add flags?
    if len(activations.size()) == 2 and len(weights.size()) == 2:
        print("Linear layer with a batch")
        # TODO how do we deal with batches?

    elif len(activations.size()) == 1 and len(weights.size()) == 2:
        print("Linear layer single entry.")
    # TODO add convolutions and more.
    else:
        raise NotImplementedError("Only linear layers currently implemented.")


def layerwise_relevance(model) -> list:
    """
    Carries out the layer-wise relevance propagation procedure
    ref
    and returns the relevance's.
    :param model:
    :return: dict The relevance values per layer as a dict of Tensors.
    """
    # TODO test for single decision only?
    # organise the data (preferably a list of alternating activations and weights)
    depth = len(model.state_dict())
    print(type(model.state_dict()))
    layers = list()
    for param, tensor in reversed(model.state_dict().items()):
        if param.find("weight") != -1:
            layers.append(torch.tensor([0.]))
            layers.append(tensor)
    for num, (layer, activation) in enumerate(reversed(model.activations.items())):
        layers[(num * 2)] = activation

    # simple sanity check to make sure the layers are in the right order.
    for item in layers:
        print(item.size())

    # TODO think about dimensions for batch operations
    # TODO think about conv dimensions.

    relevances = list()
    rel = torch.softmax(layers[0])
    # iterate through the list of layers and compute relevances.
    for i in range(0, len(layers), 2):
        rel = propagate_relevance(layers[i], layers[i+1], rel)
        relevances.append(rel)

    return relevances

