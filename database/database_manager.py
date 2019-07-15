from database.training_run import TrainingRun
from database.training_data import TrainingData


def create_database(db):
    db.connect()
    db.create_tables([TrainingRun, TrainingData])
    db.close()
