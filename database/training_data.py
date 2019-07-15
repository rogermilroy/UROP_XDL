from playhouse.postgres_ext import *
from database.base_model import BaseModel
from training_run import TrainingRun


class TrainingData(BaseModel):
    training_run = ForeignKeyField(TrainingRun)
    epoch_number = IntegerField()
    epoch_minibatch_number = IntegerField()
    total_minibatch_number = IntegerField()
    inputs = BinaryJSONField()
    model_state = BinaryJSONField()
    outputs = BlobField()
