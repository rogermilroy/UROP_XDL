from database.base_model import BaseModel
from playhouse.postgres_ext import *


class TrainingRun(BaseModel):
    model_name = CharField()
    training_run = IntegerField()
    metadata = BinaryJSONField()

    def __str__(self):
        return "<< TrainingRun. Model Name: " + self.model_name + \
               " Training Run: " + str(self.training_run) + \
               " Metadata: " + str(self.metadata) + " >>"
