from database.base_model import BaseModel
from playhouse.postgres_ext import *


class TrainingRun(BaseModel):
    model_name = CharField(unique=True)
    training_run = IntegerField()
    metadata = BinaryJSONField()
