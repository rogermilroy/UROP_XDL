from utils.network_details import get_ip
from utils.data_processing import *
from torch import Tensor
import torch
import zmq


class NNDataExtractor:
    """
    Extracts and sends NN data to Datalake.
    """
    def __init__(self):
        """
        Creates the zmq context, socket and binds to the relevant port.
        """
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.PUSH)
        self.ip = get_ip()
        self.port = 5555
        # currently only allowing tcp connection. May add more for flexibility later.
        self.sock.bind('tcp://' + str(self.ip) + ":" + str(self.port))
        print(str(self.ip) + ":" + str(self.port))
        self.total_minibatch_number = 0

    def extract_data(self, model_name: str, training_run_number: int, epoch: int,
                     epoch_minibatch: int, inputs: Tensor, model_state: dict, outputs: Tensor, targets: Tensor) -> None:
        """
        TODO add link to training run.
        Wrapper that encodes and sends data. Auto increments total minibatch number.
        :param epoch: int Number of times through dataset.
        :param epoch_minibatch: int The minibatch number within current epoch
        :param inputs: Tensor The inputs as a Tensor  # TODO link to original dataset.
        :param model_state: dict The parameters of the model as a dict of Tensors.
        :param outputs: Tensor The outputs of the model as a Tensor
        :param targets: Tensor The target values of the model as a Tensor.
        :return: None
        """
        self.send_json(data_to_json(model_name=model_name,
                                    training_run_number=training_run_number, epoch=epoch,
                                    epoch_minibatch=epoch_minibatch,
                                    total_minibatch=self.total_minibatch_number,
                                    inputs=inputs, model_state=model_state, outputs=outputs,
                                    targets=targets))
        self.total_minibatch_number += 1

    def extract_metadata(self, model_name: str, training_run_number: int, epochs: int,
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
        self.send_json(metadata_to_json(model_name=model_name,
                                        training_run_number=training_run_number, epochs=epochs,
                                        batch_size=batch_size,
                                        cuda=cuda, model=model, criterion=criterion,
                                        optimizer=optimizer, metadata=metadata))

    def send_json(self, json_data: bytes) -> None:
        """
        Sends the data to the socket to be sent.
        Catches sending errors and prints them to console for debugging.
        :param json_data: JSON encoded data to be sent.
        :return: None
        """
        try:
            self.sock.send(json_data)
        except zmq.ZMQError as e:
            print(e)


if __name__ == '__main__':
    test_extractor = NNDataExtractor()
    test_extractor.send_json(b'Testing.')
