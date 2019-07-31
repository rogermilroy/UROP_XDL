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


def data_to_json(model_name: str, training_run_number: int, epoch: int, epoch_minibatch: int,
                 total_minibatch: int, inputs: Tensor,
                 model_state: dict, outputs: Tensor, targets: Tensor) -> bytes:
    """
    Takes raw data from the training loop and converts to JSON to send to datalake.
    :param model_name: str The name of the model
    :param training_run_number: int The number of the training run.
    :param epoch: int Number of times through dataset.
    :param epoch_minibatch: int The minibatch number within current epoch
    :param total_minibatch: int The total number of minibatches.
    :param inputs: Tensor The inputs as a Tensor  # TODO link to original dataset.
    :param model_state: dict The parameters of the model as a dict of Tensors.
    :param outputs: Tensor The outputs of the model as a Tensor
    :param targets: Tensor The target values of the model as a Tensor.
    :return: bytes JSON encoded dict. 
    """
    data = dict()  # TODO add training run data.
    data['model_name'] = model_name
    data['training_run_number'] = training_run_number
    data['epoch'] = epoch
    data['epoch_minibatch'] = epoch_minibatch
    data['total_minibatch'] = total_minibatch
    data['inputs'] = encode_tensor(inputs)
    data['outputs'] = encode_tensor(outputs)
    data['targets'] = encode_tensor(targets)

    data['model_state'] = encode_model_state(model_state)

    return json.dumps(data)


def metadata_to_json(model_name: str, training_run_number: int, epochs: int, batch_size: int,
                     cuda: bool, model: torch.nn.Module, criterion,
                     optimizer: torch.optim.Optimizer, metadata: dict) -> bytes:
    """
    Converts metadata to json.
    :param model_name: str The name of the model
    :param training_run_number: int The number of the training run.
    :param epochs: int The number of epochs to be run. (Full runs through the dataset)
    :param batch_size: int The number of samples per minibatch.
    :param cuda: bool Whether the training is on a CUDA GPU
    :param model: torch.Module The model being trained.
    :param criterion: torch.nn.Criterion The loss criterion.
    :param optimizer: torch.optim.Optimizer The optimisation algorithm used.
    :param metadata: dict Any other metadata to be stored.
    :return: bytes JSON encoded dict.
    """
    mdata = dict()
    mdata['model_name'] = model_name
    mdata['training_run_number'] = training_run_number
    mdata['epochs'] = epochs
    mdata['batch_size'] = batch_size
    mdata['cuda'] = cuda
    mdata['model'] = str(model)
    mdata['criterion'] = str(criterion)
    mdata['optimizer'] = str(optimizer)
    mdata['metadata'] = metadata

    return json.dumps(mdata)
