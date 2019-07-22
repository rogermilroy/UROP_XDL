from utils.network_details import get_ip
from utils.data_processing import *
import zmq



class NNDataExtractor:
    """
    Extracts and sends NN data to Datalake.
    """
    def __init__(self):
        """
        Creates the zmq context, socket and binds to the relevant port.
        """
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.ip = get_ip()
        self.port = 50005
        # currently only allowing tcp connection. May add more for flexibility later.
        self.socket.bind('tcp://' + str(self.ip) + ":" + str(self.port))
        self.total_minibatch_number = 0

    def extract_data(self, epoch, epoch_minibatch, inputs, model_state, outputs, targets):
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
        self.send_json(data_to_json(epoch, epoch_minibatch, self.total_minibatch_number, inputs,
                                    model_state, outputs, targets))
        self.total_minibatch_number += 1

    def extract_metadata(self, model_name, training_run_number, epochs, batch_size, cuda, model,
                         criterion, optimizer, metadata):
        self.send_json(metadata_to_json(model_name, training_run_number, epochs, batch_size,
                                        cuda, criterion, optimizer, metadata))

    def send_json(self, json_data):
        """
        Sends the data to the socket to be sent.
        Catches sending errors and prints them to console for debugging.
        :param json_data: JSON encoded data to be sent.
        :return: None
        """
        try:
            self.socket.send(json_data)
        except zmq.ZMQError as e:
            print(e)
