from database.database.base_model import BaseModel
from playhouse.postgres_ext import *


class TrainingRun(BaseModel):
    model_name = CharField()
    training_run = IntegerField()
    epochs = IntegerField()
    batch_size = IntegerField()
    cuda = BooleanField()
    model = CharField()
    criterion = CharField()
    optimizer = CharField()
    metadata = BinaryJSONField()

    def __str__(self):
        return "<< TrainingRun. Model Name: " + self.model_name + \
               " Training Run: " + str(self.training_run) + \
               " Epochs: " + str(self.epochs) + \
               " Batch Size: " + str(self.batch_size) + \
               " On CUDA: " + str(self.cuda) + \
               " Model: " + self.model + \
               " Criterion: " + self.criterion + \
               " Optimizer: " + self.optimizer + \
               " Metadata: " + str(self.metadata) + " >>"
