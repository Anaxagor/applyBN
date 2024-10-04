from applybn.data_generation.base_class import DataGenerator
from bamt.networks.base import BaseNetwork
from pandas import DataFrame, concat
from typing import Union
from applybn.core import copy_data


class BNClassBalance(DataGenerator):
    """
    Class to balance unbalanced data.

    Methods:
        balance:
    """
    def __init__(
            self,
            bn: BaseNetwork
    ):
        super().__init__(bn)

    @copy_data
    def balance(self,
                unbalanced_data: DataFrame,
                class_column: Union[str, None],
                strategy: Union[str, int],
                shuffle: bool,
                ):
        """

        Args:
            unbalanced_data: data with unbalanced classes
            class_column: name of class column in data
            strategy: whether all class sizes are equal to currently maximum class (str "max_class") size
            or manually chosen size (int n)
            shuffle: whether returned data is shuffled or not

        Returns:
            balanced data

        """
        class_column = class_column if class_column is not None else unbalanced_data.columns[-1]
        classes_size = unbalanced_data.value_counts(unbalanced_data[class_column], sort=True)

        if strategy == 'max_class':
            max_class_size = classes_size[0]
        else:
            max_class_size = strategy

        additional_data_lst = []
        for cur_class in unbalanced_data[class_column].unique():
            cur_class_size = classes_size[cur_class]
            need_size = 0 if max_class_size - cur_class_size < 0 else max_class_size - cur_class_size

            additional_data_lst.append(self.bn.sample(evidence={class_column: cur_class},
                                                      n=need_size))

        balanced_data = concat([unbalanced_data] + additional_data_lst)

        if shuffle:
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

        return balanced_data
