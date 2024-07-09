# TODO: implement train_test_expand functions
from typing import Union

import numpy as np
import pandas as pd

from applybn.core import copy_data

@copy_data
def train_test_expand(data: Union[pd.DataFrame, np.ndarray], **parameters):
    """
    Function to expand the train and test data
    :param data:
    :param parameters:
    :return:
    """
    pass
