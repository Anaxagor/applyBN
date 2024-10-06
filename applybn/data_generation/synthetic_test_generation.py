from applybn.data_generation.base_class import DataGenerator
from sklearn.exceptions import NotFittedError
from pandas import DataFrame, concat
from typing import Union, Callable
from applybn.core import copy_data
from sklearn.metrics import f1_score
from numpy import inf
import numpy as np


class BNTestGenerator(DataGenerator):
    def __init__(
            self,
    ):
        super().__init__()

    @staticmethod
    def evaluate_data(data, clf, metric, n_trials=100):
        res_metric = np.zeros(n_trials)
        for i in range(n_trials):
            shuffled_data = data.sample(frac=1).reset_index(drop=True)
            x, y = shuffled_data[data.columns[:-1]], shuffled_data[data.columns[-1]]

            size = len(data)
            x_train, y_train = x.iloc[:int(0.7 * size)], y.iloc[:int(0.7 * size)]
            x_test, y_test = x.iloc[int(0.7 * size):], y.iloc[int(0.7 * size):]
            clf.fit(x_train, y_train)

            res_metric[i] = metric(y_test, clf.predict(x_test))

        return res_metric

    def generate_test_data(self,
                           real_data: DataFrame,
                           clf,
                           one_sample_size: int = 1000,
                           n_sample_try: int = 10,
                           metric: Callable = f1_score,
                           accept_threshold: Union[float, None] = None):

        if self.bn is None:
            raise NotFittedError('BN must be fitted firstly')

        synth_test_data = []
        real_metric = self.evaluate_data(real_data, clf, metric)
        real_metric_mean = real_metric.mean()

        accept_threshold = inf if accept_threshold is None else accept_threshold

        for i in range(n_sample_try):
            cur_synth_data = self.bn.sample(one_sample_size)[real_data.columns].astype(real_data.dtypes.to_dict())
            cur_synth_metric = self.evaluate_data(cur_synth_data, clf, metric).mean()

            if abs(real_metric_mean - cur_synth_metric) / real_metric_mean <= accept_threshold:
                synth_test_data.append(cur_synth_data)
                print('Accepted')

        synth_test_data = concat(synth_test_data).reset_index(drop=True)
        
        return synth_test_data






