from bamt.networks.base import BaseNetwork
from pandas import DataFrame
from applybn.core import copy_data
from applybn.data_generation.get_bn import get_hybrid_bn


class DataGenerator:
    """
    Base class for data generation methods

    Attributes:
        bn (BaseNetwork): Bayesian Network for synthetic data generation.
    """

    def __init__(
            self,
    ):
        self.bn = None

    # @copy_data
    def fit(self, data: DataFrame):
        self.bn = get_hybrid_bn(data)
