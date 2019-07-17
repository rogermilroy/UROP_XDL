from playhouse.postgres_ext import *
from database.base_model import BaseModel
from database.training_run import TrainingRun
from io import BytesIO
import numpy as np
import torch


class TrainingData(BaseModel):
    training_run = ForeignKeyField(TrainingRun)
    epoch_number = IntegerField()
    epoch_minibatch_number = IntegerField()
    total_minibatch_number = IntegerField()
    inputs = BinaryJSONField()
    model_state = BinaryJSONField()
    outputs = BinaryJSONField()
    targets = BinaryJSONField()

    def __str__(self):
        self.inputs = torch.tensor(np.load(BytesIO(self.inputs)))
        return "<< TrainingData, TrainingRun: " + str(self.training_run) + \
               " \nEpoch: " + str(self.epoch_number) + \
               " \nMinibatch: " + str(self.epoch_minibatch_number) + \
               " \nTotal Minibatch: " + str(self.total_minibatch_number) + \
               " \nInputs: " + str(self.inputs) + \
               " \nModel State: " + str(self.model_state) + \
               " \nOutputs: " + str(self.outputs) + \
               " \nTargets: " + str(self.targets) + " >>"
