from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class FeatureGenerator(ABC):
    """
    Abstract class for feature generation
    """

    def __init__(self, **parameters):
        self._parameters = parameters

    @abstractmethod
    def generate_features(
        self, data: Union[pd.DataFrame, np.ndarray], target
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Abstract method to generate features
        :param data:
        :param target:
        :return:
        """
        pass
