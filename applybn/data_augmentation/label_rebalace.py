# TODO: Implement label rebalance tool
from typing import Union

import numpy as np
import pandas as pd

from applybn.core import copy_data


@copy_data
def label_rebalance(data: Union[pd.DataFrame, np.ndarray], **parameters):
    """
    Function to rebalance the labels
    :param data:
    :param parameters:
    :return:
    """
    pass
