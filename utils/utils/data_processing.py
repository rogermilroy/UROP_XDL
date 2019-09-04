from torch import Tensor
import torch
import numpy as np
import ujson as json
# import zmq.utils.jsonapi as json


def encode_model_state(model_state: dict) -> dict:
    """
    Utility to encode the model state as a dict of dicts to enable json encoding.
    :param model_state: dict containing the weight and bias Tensors.
    :return: dict containing weights and biases encoded as dicts.
    """
    state = dict()
    for key, value in model_state.items():
        rkey = key.replace('.', '_dot_')
        state[rkey] = encode_tensor(value)
    return state


def decode_model_state(model_state: dict) -> dict:
    """
    Utility to decode the model state back to a dict of Tensors.
    :param model_state: dict containing weights and biases as dicts.
    :return: dict containing weight and bias Tensors.
    """
    state = dict()
    for key, value in model_state:
        rkey = key.replace('_dot_', '.')
        state[rkey] = decode_tensor(value)
    return state


def encode_tensor(tensor: Tensor) -> dict:
    """
    Utility for converting Torch Tensors to dict's for encoding to JSON.
    :param tensor: Torch Tensor.
    :return: dict containing tensor data and metadata.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("Must be a Torch.Tensor.")
    t_dict = dict()
    t_dict["device"] = str(tensor.device)
    t_dict["dtype"] = str(tensor.dtype)
    t_dict["layout"] = str(tensor.layout)
    t_dict["grad"] = encode_tensor(tensor.grad) if tensor.grad is not None else tensor.grad
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.device == torch.device("cuda"):
        tensor.cpu()
    if t_dict["layout"] == "torch.sparse_coo":
        tensor = tensor.to_dense()
    t_dict["data"] = tensor.numpy().tolist()
    return t_dict


def decode_tensor(tensor_dict: dict) -> Tensor:
    """
    Utility to convert dicts to Torch Tensor.
    Issues are that we cannot necessarily return to original device and we can't easily return to
    original layout.
    TODO construct sparse Tensor from dense tensor.
    :param tensor_dict: A dict constructed by encode_tensor.
    :return: Torch Tensor.
    """
    if not isinstance(tensor_dict, dict):
        raise TypeError("Must be a dict.")
    tens = torch.tensor(np.array(tensor_dict["data"])).to(tensor_dict["dtype"])
    if tensor_dict["device"] == str(torch.device("cuda")) and torch.cuda.is_available():
        tens = tens.to(tensor_dict["device"])
    # TODO reconstruct sparse tensor from dense.
    # TODO reattach gradient (Might not be possible)
    return tens
