from typing import Union

import numpy as np
import pandas as pd

from applybn.core import copy_data
from feature_generator import FeatureGenerator


class ProbabilisticFeatureGenerator(FeatureGenerator):
    """
    Class to generate features based on probabilistic methods
    """

    def __init__(self, **parameters):
        super().__init__(**parameters)

    @copy_data  # copying data is crucial to avoid modifying the original data
    def generate_features(
        self, data: Union[pd.DataFrame, np.ndarray], target
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Abstract method to generate features
        :param data:
        :param target:
        :return:
        """
        # TODO: Implement the feature generation logic here
        return data
