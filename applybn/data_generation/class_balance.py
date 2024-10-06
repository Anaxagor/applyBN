from applybn.data_generation.base_class import DataGenerator
from pandas import DataFrame, concat
from typing import Union
from sklearn.exceptions import NotFittedError
from applybn.core import copy_data


class BNClassBalancer(DataGenerator):
    """
    Class to balance disbalanced data.

    Methods:
        balance:
    """
    def __init__(
            self,
    ):
        super().__init__()

    # @copy_data
    def balance(self,
                disbalanced_data: DataFrame,
                class_column: Union[str, None] = None,
                strategy: Union[str, int] = 'max_class',
                shuffle: bool = True,
                ):
        """

        Args:
            disbalanced_data: data with disbalanced classes
            class_column: name of class column in data
            strategy: whether all class sizes are equal to currently maximum class (str "max_class") size
            or manually chosen size (int n)
            shuffle: whether returned data is shuffled or not

        Returns:
            balanced data

        """
        if self.bn is None:
            raise NotFittedError('BN must be fitted firstly')

        class_column = class_column if class_column is not None else disbalanced_data.columns[-1]
        classes_size = disbalanced_data.value_counts(disbalanced_data[class_column], sort=True)

        if strategy == 'max_class':
            max_class_size = classes_size[0]
        else:
            max_class_size = strategy

        additional_data_lst = []
        for cur_class in disbalanced_data[class_column].unique():
            cur_class_size = classes_size[cur_class]
            need_size = 0 if max_class_size - cur_class_size < 0 else max_class_size - cur_class_size

            if need_size:
                additional_data_lst.append(self.bn.sample(need_size, evidence={'label': 0})[disbalanced_data.columns])

        balanced_data = concat([disbalanced_data] + additional_data_lst)

        if shuffle:
            balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

        return balanced_data[disbalanced_data.columns].astype(disbalanced_data.dtypes.to_dict())
