from copy import deepcopy

from database.database_manager import DatabaseManager


class NNDataExtractor:
    """
    Class for extracting all the training data for later analysis.
    """
    def __init__(self):
        self.db = DatabaseManager
        self.total_epoch_number = 0

    def extract_metadata(self, model_name, training_run_number, metadata):
        # TODO figure out exactly what metadata we want/need
        return

    def extract_data(self, epoch, epoch_minibatch, inputs, model_state, outputs, targets):
        """
        This method will send extracted data to the Database
        :param epoch:
        :param epoch_minibatch:
        :param inputs:
        :param model_state:
        :param outputs:
        :param targets:
        :return:
        """
        epoch = epoch
        epoch_minibatch = epoch_minibatch
        total_minibatch = self.total_epoch_number
        # increment total epoch number.
        self.total_epoch_number += 1
        inp = deepcopy(inputs)
        model_state = model_state
        out = deepcopy(outputs)
        tar = deepcopy(targets)
        return epoch, epoch_minibatch, inp, model_state, out, tar
