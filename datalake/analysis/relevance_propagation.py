import torch
import torch.nn.functional as func
from analysis.analysis_utils import apply_rho
from torch import Tensor


def linear_alpha_beta_relevance(activations: Tensor, weights: Tensor, relevance: Tensor, alpha,
                                beta) -> Tensor:
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


def layer_relevance(activations: Tensor, weights, relevances: Tensor, rho: func = lambda
        x: x) -> Tensor:
    """
    Generic layerwise relevance rule. Suitable for conv, dense and pooling (with thought) layers.
    Use input rule for input layers.

    The regular rule is the default rho (the identity function)
    To use the epsilon rule use rho = lambda x : x + epsilon * torch.clamp(x, min=0.);

    Written with reference to https://towardsdatascience.com/indepth-layer-wise-relevance-propagation-340f95deb1ea
    Explainable AI: Interpreting, Explaining and Visualizing Deep Learning and
    http://www.heatmapping.org/tutorial/

    :param activations: The activations for the layer
    :param weights: The layers weights
    :param relevances: The prior layers relevance (to be propagated through this layer)
    :param rho: The function that allows the epsilon rule or other rules to be implemented.
    :return:
    """
    z = apply_rho(weights, rho).forward(activations)
    s = (relevances / (z + 1e-9)).data
    (z * s).sum().backward()
    c = activations.grad.data
    return activations.detach() * c


def input_relevance(activations: Tensor, weights, relevances: Tensor, inputs: Tensor) -> Tensor:
    """
    Layer for propagating relevance to the inputs to the network.
    z beta rule
    w^2 rule to follow
    :param activations:
    :param weights:
    :param relevances:
    :return:
    """

    in_min = torch.min(inputs)
    in_max = torch.max(inputs)

    lb = ((activations.data * 0 + in_min).data).requires_grad_(True)
    hb = ((activations.data * 0 + in_max).data).requires_grad_(True)

    z = weights.forward(activations) + 1e-9
    z -= apply_rho(weights, rho=lambda p: p.clamp(min=0)).forward(lb)
    z -= apply_rho(weights, rho=lambda p: p.clamp(max=0)).forward(hb)
    s = (relevances / z).data  # step 2
    (z * s).sum().backward()
    c, cp, cm = activations.grad, lb.grad, hb.grad
    return (activations * c + lb * cp + hb * cm).data


# at each layer we need to decide the shape/type of the layer.
def propagate_relevance(activations: Tensor, weights, relevances: Tensor) -> \
        Tensor:
    """
    Wrapper to decide which kind of relevance propagation is needed.
    Chooses between epsilon (to absorb some contradictory/spurious attribution)
    gamma (to accentuate the positive evidence) and input for the final layer before the inputs.
    Also w^2 but that is yet to be done.
    :param activations: Tensor of activations.
    :param weights: Tensor of weights
    :param relevances: Tensor of relevances
    :return: Tensor of relevance values
    """
    # TODO redo these to instead examine the type of each layer.
    # TODO think if these make more sense in the main loop (have a reference for the depth of the
    #  layers...

    # print(type(weights))
    # print(torch.min(activations))
    if False:
        #TODO figure out how to separate layers under TanH activation function.....
        return input_relevance(activations=activations, weights=weights, relevances=relevances,
                               inputs=activations)
    else:
        # print('using regular')
        return layer_relevance(activations=activations, weights=weights, relevances=relevances,
                           rho=lambda x: x + (1.0 * x.clamp(min=0.)))

    # check linear layer TODO add flags?
    # if len(activations.size()) == 2 and len(weights.size()) == 2:
    #     # print("Linear layer with a batch")
    #     return linear_alpha_beta_relevance(activations=activations, weights=weights,
    #                                        relevance=relevances,
    #                                        alpha=1.0, beta=0.0)
    #     # TODO how do we deal with batches?
    #
    # elif len(activations.size()) == 1 and len(weights.size()) == 2:
    #     # print("Linear layer single entry.")
    #     return layer_relevance(activations=activations, weights=weights, relevances=relevances,
    #                            rho=lambda x: x + 0.25 * torch.clamp(x, min=0.))
    # # TODO add convolutions and more.
    # else:
    #     raise NotImplementedError("Only linear layers currently implemented.")


def layerwise_relevance(model, inputs) -> tuple:
    """
    Carries out the layer-wise relevance propagation procedure
    ref
    and returns the relevance's.
    Assumptions: 
    1. The model is in the state immediately after the decision is made. 
    This is to preserve the activations.
    2. The model used is using the pattern established of storing activations
    per layer in an ordered dict called activations.
    3. The model is for classification.
    :param model: The model (Neural Network)
    :param inputs: The inputs to the model.
    :return: dict The relevance values per layer as a dict of Tensors.
    """
    # TODO test for single decision only?
    # organise the data (preferably a list of alternating activations and weights)

    # New way

    # Old way
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
    # for item in layers:
    #     print(item.size())

    # TODO think about dimensions for batch operations
    # TODO think about conv dimensions.

    relevances = list()
    # this is to establish the outputs from the activations.
    layers[0] = func.softmax(layers[0])
    mask = torch.zeros_like(layers[0])
    mask[0, torch.argmax(layers[0])] = 1.
    rel = layers[0] * mask
    relevances.append(rel)
    # iterate through the list of layers and compute relevances.
    print(range(1, len(layers) - 1, 2))
    for i in range(1, len(layers) - 1, 2):
        rel = propagate_relevance(activations=layers[i + 1], weights=layers[i], relevances=rel)
        relevances.append(rel)

    return relevances, weights


def new_layerwise_relevance(model, inputs, index):
    """
    New layerwise relevance implementation using new method described in
    Explainable AI: Interpreting, Explaining and Visualizing Deep Learning and
    http://www.heatmapping.org/tutorial/

    :param model: The model over which the explanation will be done
    :param inputs: The original inputs
    :param index: The rank of the result desired (eg largest probability predicted or 2nd etc)
    :return: The relevances for each layer of the model.
    """
    weights = list()
    for param, tensor in reversed(model.state_dict().items()):
        if param.find("weight") != -1:
            weights.append(tensor)

    model_layers = list(model._modules.values())
    L = len(model_layers)

    # run model layers and collect activations
    relevances = list()
    activs = [inputs] + [None] * L  #TODO changn
    for l in range(L):
        activs[l + 1] = model_layers[l].forward(activs[l])

    # create the mask to select the relevance desired
    probs = torch.softmax(torch.sigmoid(activs[-1]), dim=1)
    # print(probs)
    mask = torch.zeros_like(probs)
    top_k = torch.topk(probs, k=probs.shape[1]).indices
    # print(top_k[0, index])
    mask[0, top_k[0, index]] = 1.
    rel = probs * mask
    relevances.append(rel.data)

    # propagate the relevance through the layers
    for i in reversed(range(1, L)):
        activs[i] = (activs[i].data).requires_grad_(True)
        rel = propagate_relevance(activations=activs[i], weights=model_layers[i], relevances=rel)
        relevances.append(rel)

    # here special case because of inputs.
    activs[0] = (activs[0].data).requires_grad_(True)
    rel = input_relevance(activations=activs[0], weights=model_layers[0], relevances=rel,
                          inputs=inputs)

    return rel, weights
