from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import Tensor


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
    for key, value in model_state.items():
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
    tens = torch.tensor(np.array(tensor_dict["data"]))#.to(tensor_dict["dtype"]) TODO fix this.
    if tensor_dict["device"] == str(torch.device("cuda")) and torch.cuda.is_available():
        tens = tens.to(tensor_dict["device"])
    # TODO reconstruct sparse tensor from dense.
    # TODO reattach gradient (Might not be possible)
    return tens


def encode_relevance(relevance: List[Tensor]) -> List[dict]:
    """
    Encodes relevance list (layers for one relevance)
    :param relevance:
    :return:
    """
    encoded_relevance = list()
    for layer in relevance:
        encoded_relevance.append(encode_tensor(layer))
    return encoded_relevance


def decode_relevance(encoded_relevance: List[dict]) -> List[Tensor]:
    """
    Decodes relevance list (layers for one relevance)
    :param encoded_relevance:
    :return:
    """
    decoded_relevance = list()
    for layer in encoded_relevance:
        decoded_relevance.append(decode_tensor(layer))
    return decoded_relevance


def encode_relevances(relevances: List[List[Tensor]]) -> List[List[dict]]:
    """
    Encodes multiple relevances as list of lists of dicts that can be stored in Mongo
    :param relevances:
    :return:
    """
    encoded_relevances = list()
    for relevance in relevances:
        encoded_relevances.append(encode_relevance(relevance))
    return encoded_relevances


def decode_relevances(encoded_relevances: List[List[dict]]) -> List[List[Tensor]]:
    """
    Decodes an encoded relevances list. (Multiple relevances)
    :param encoded_relevances:
    :return:
    """
    decoded_relevances = list()
    for encoded_relevance in encoded_relevances:
        decoded_relevances.append(decode_relevance(encoded_relevance))
    return decoded_relevances


def encode_diffs(diffs: Dict[int, Tuple[List[Tensor], List[Tensor], Tensor, Tensor, Tensor,
                                        Tensor]]) -> \
        Dict[str, Tuple[List[dict], List[dict], dict, dict, dict, dict]]:
    """
    Encodes diff list for storage in Mongo.
    :param diffs:
    :return:
    """
    encoded_diffs = dict()
    for key, (rel0, rel1, top3_before, top3_after, outs_before, outs_after) in diffs.items():
        encoded_diffs[str(key)] = (
            encode_relevance(rel0),
            encode_relevance(rel1),
            encode_tensor(top3_before),
            encode_tensor(top3_after),
            encode_tensor(outs_before),
            encode_tensor(outs_after)
        )
    return encoded_diffs


def decode_diffs(diffs: Dict[str, Tuple[List[dict], List[dict], dict, dict, dict, dict]]) -> \
        Dict[int, Tuple[List[Tensor], List[Tensor], Tensor, Tensor, Tensor, Tensor]]:
    """
    Encodes diff list for storage in Mongo.
    :param diffs:
    :return:
    """
    decoded_diffs = dict()
    for key, (
    rel0, rel1, top3_before, top3_after, outs_before, outs_after) in diffs.items():
        decoded_diffs[int(key)] = (
            decode_relevance(rel0),
            decode_relevance(rel1),
            decode_tensor(top3_before),
            decode_tensor(top3_after),
            decode_tensor(outs_before),
            decode_tensor(outs_after)
        )
    return decoded_diffs
