from database.training_run import TrainingRun
from database.training_data import TrainingData
from io import BytesIO
import numpy as np
from utils.tensor_processing import *


class DatabaseManager(object):
    """
    Class to wrap database interactions.
    """
    def __init__(self, db):
        self.db = db
        self.training_run = None

    def create_database(self):
        self.db.connect()
        self.db.create_tables([TrainingRun, TrainingData])
        self.db.close()

    def save_metadata(self, model_name, training_run, metadata):
        self.db.connect()
        self.training_run = TrainingRun.create(model_name=model_name, training_run=training_run,
                                               metadata=metadata)
        print(self.training_run)

        self.db.close()

    def save_training_data(self, epoch, epoch_minibatch, tot_minibatch, inputs,
                           model_state, outputs, targets):
        # with BytesIO() as b:
        #     np.save(b, inputs)
        #     ser_in = b.getvalue()
        #     np.save(b, outputs)
        #     ser_out = b.getvalue()
        #     np.save(b, targets)
        #     ser_tar = b.getvalue()

        model_state = encode_model_state(model_state)

        self.db.connect()
        new_data = TrainingData.create(training_run=self.training_run, epoch_number=epoch,
                                       epoch_minibatch_number=epoch_minibatch,
                                       total_minibatch_number=tot_minibatch,
                                       inputs=encode_tensor(inputs),
                                       model_state=model_state, outputs=encode_tensor(outputs),
                                       targets=encode_tensor(targets))
        new_data.save()
        self.db.close()
