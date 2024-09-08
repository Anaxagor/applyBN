from base_class import DataGenerator
from bamt.networks.base import BaseNetwork
from pandas import DataFrame, concat
from typing import Union
from applybn.core import copy_data


class BNModelTesting(DataGenerator):
    def __init__(self,
                 models: list,
                 bn: BaseNetwork):
        super().__init__(bn)

    def test_models(self):
        ...


