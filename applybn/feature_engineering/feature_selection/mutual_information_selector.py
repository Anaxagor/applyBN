from typing import Union, List

import numpy as np
import pandas as pd

from applybn.core import copy_data
from feature_selector import FeatureSelector


class MIFeatureSelector(FeatureSelector):
    """
    Class for feature selection based on mutual information
    """

    def __init__(self, **parameters):
        super().__init__(**parameters)

    @copy_data  # copying data is crucial to avoid modifying the original data
    def select_features(
        self, data: Union[pd.DataFrame, np.ndarray], target: Union[str, int]
    ) -> List[Union[str, int]]:
        """
        Abstract method to select features
        :param data:
        :param target:
        :return:
        """

        dummy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # TODO: Implement feature selection using mutual information
        return dummy
