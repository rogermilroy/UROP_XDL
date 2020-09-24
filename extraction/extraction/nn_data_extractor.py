import torch
import ujson as json
import zmq
from torch import Tensor
from utils.data_processing import *
from utils.network_details import get_ip


class NNDataExtractor:
    """
    Extracts and sends NN data to Datalake.
    """
    def __init__(self, model_name: str, training_run_number: int, local=False):
        """
        Creates the zmq context, socket and binds to the relevant port.
        """
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.PUSH)
        if not local:
            self.ip = get_ip()
        else:
            self.ip = '127.0.0.1'
        self.port = 5555
        # currently only allowing tcp connection. May add more for flexibility later.
        self.sock.bind('tcp://' + str(self.ip) + ":" + str(self.port))
        print(str(self.ip) + ":" + str(self.port))
        self.total_minibatch_number = 0
        self.model_name = model_name
        self.training_run_number = training_run_number

    def extract_data(self,  epoch: int,
                     epoch_minibatch: int, inputs: Tensor, model_state: dict, outputs: Tensor, targets: Tensor) -> None:
        """
        Encodes and sends data. Auto increments total minibatch number.
        :param epoch: int Number of times through dataset.
        :param epoch_minibatch: int The minibatch number within current epoch
        :param inputs: Tensor The inputs as a Tensor  # TODO link to original dataset.
        :param model_state: dict The parameters of the model as a dict of Tensors.
        :param outputs: Tensor The outputs of the model as a Tensor
        :param targets: Tensor The target values of the model as a Tensor.
        :return: None
        """
        data = dict()
        data['model_name'] = self.model_name
        data['training_run_number'] = self.training_run_number
        data['epoch'] = epoch
        data['epoch_minibatch'] = epoch_minibatch
        data['total_minibatch'] = self.total_minibatch_number
        data['inputs'] = encode_tensor(inputs)
        data['outputs'] = encode_tensor(outputs)
        data['targets'] = encode_tensor(targets)

        data['model_state'] = encode_model_state(model_state)

        self.send_json(json.dumps(data))
        self.total_minibatch_number += 1

    def extract_metadata(self, epochs: int,
                         batch_size: int, cuda: bool, model: torch.nn.Module,
                         criterion, optimizer: torch.optim.Optimizer,
                         metadata: dict) -> None:
        """
        Sends the metadata to the socket to be sent.
        :param model_name: str The name of the model
        :param training_run_number: int The number of the training run.
        :param epochs: int The number of epochs to be run. (Full runs through the dataset)
        :param batch_size: int The number of samples per minibatch.
        :param cuda: bool Whether the training is on a CUDA GPU
        :param model: torch.nn.Module The model being trained.
        :param criterion: torch.nn.Criterion The loss criterion.
        :param optimizer: torch.optim.Optimizer The optimisation algorithm used.
        :param metadata: dict Any other metadata to be stored.
        :return: None
        """
        mdata = dict()
        mdata['model_name'] = self.model_name
        mdata['training_run_number'] = self.training_run_number
        mdata['epochs'] = epochs
        mdata['batch_size'] = batch_size
        mdata['cuda'] = cuda
        mdata['model'] = str(model)
        mdata['criterion'] = str(criterion)
        mdata['optimizer'] = str(optimizer)
        mdata['metadata'] = metadata

        self.send_json(json.dumps(mdata))

    def extract_final_state(self, final_epochs, final_model_state, best_params_epoch):
        mdata = dict()
        mdata['model_name'] = self.model_name
        mdata['training_run_number'] = self.training_run_number
        mdata['final_epochs'] = final_epochs
        mdata['final_model_state'] = encode_model_state(final_model_state)
        mdata['best_params_epoch'] = best_params_epoch
        print("sending final state...")
    
        self.send_json(json.dumps(mdata))

    def send_json(self, json_data: bytes) -> None:
        """
        Sends the data to the socket to be sent.
        Catches sending errors and prints them to console for debugging.
        :param json_data: JSON encoded data to be sent.
        :return: None
        """
        try:
            self.sock.send_string(json_data)
        except zmq.ZMQError as e:
            print(e)


if __name__ == '__main__':
    test_extractor = NNDataExtractor("test", 0)
    test_extractor.send_json(b'Testing.')
