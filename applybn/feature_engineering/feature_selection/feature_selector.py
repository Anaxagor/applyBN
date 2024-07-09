from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd


class FeatureSelector(ABC):
    """
    Abstract class for feature selection
    """

    # any function or attribute that will be common to child classes can be defined here
    def __init__(self, **parameters):
        self._parameters = parameters

    @abstractmethod
    def select_features(
        self, data: Union[pd.DataFrame, np.ndarray], target: Union[str, int]
    ) -> List[Union[str, int]]:
        """
        Abstract method to select features
        :param data:
        :param target:
        :return:
        """
        pass
