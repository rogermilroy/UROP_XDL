from playhouse import *
from database.training_run import TrainingRun
from database.training_data import TrainingData
from io import BytesIO
import numpy as np
from torch import Tensor


def test_select_run():
    test_run = TrainingRun.get_by_id(1)
    print(test_run)


def test_select_data():
    test_data = TrainingData.get_by_id(2)
    print(test_data)


if __name__ == '__main__':
    test_select_run()
    # test_select_data()