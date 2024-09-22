import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import stats

class PartialCorrelation:
    '''
    Partial Correlation test when causal links have linear dependency
    '''
    def __init__(self):
        pass

    def run_test(self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None) -> Tuple[float, float]:
        '''
        compute the test statistics and pvalues

        :param x: input data for x
        :param y: input data for y
        :param z: input data for z
        
        :return: Returns a tuple of 2 floats-- test statistic and the corresponding pvalue
        '''
        self.x = x
        self.y = y
        self.z = z

        test_stat = self.get_correlation()
        pvalue = self.get_pvalue(test_stat)
        return test_stat, pvalue

    def get_correlation(self) -> float:
        '''
        pearson's correlation between residuals
        '''
        x_residual = self._get_residual_error(self.x).ravel()
        y_residual = self._get_residual_error(self.y).ravel()
        val, _ = stats.pearsonr(x_residual, y_residual)
        return val

    def _get_residual_error(self, v: np.ndarray) -> np.ndarray:
        
        if self.z is not None:
            beta_hat = np.linalg.lstsq(self.z, v, rcond=None)[0]
            mean = np.dot(self.z, beta_hat)
            resid = v - mean
        else:
            resid = v
        return resid

    def get_pvalue(self, value: float) -> float:
        
        dim = 2 + self.z.shape[1] if self.z is not None else 2
        deg_freedom = self.x.shape[0] - dim

        if deg_freedom < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            t_score = value * np.sqrt(deg_freedom / (1. - value * value))
            pval = stats.t.sf(np.abs(t_score), deg_freedom) * 2

        return pval

class GrowShrink:
    def __init__(self, data: pd.DataFrame, CI_test: PartialCorrelation):
        """
        Инициализация алгоритма Grow-Shrink.
        
        :param data: Табличные данные в виде pandas DataFrame.
        :param CI_test: Объект класса, реализующего условный независимый тест.
        """
        self.data = data
        self.CI_test = CI_test
    
    def run(self, target_var: str, pvalue_thres: float) -> List[str]:
        """
        Запуск алгоритма.
        
        :param target_var: Имя целевой переменной.
        :param pvalue_thres: Порог значимости p-value.
        
        :return: Список признаков, входящих в Марковское окружение.
        """
        S = set()  
        remaining_vars = set(self.data.columns) - {target_var}
        
        # Фаза Grow
        for var in remaining_vars:
            pval = self._conditional_independence_test(target_var, var, S)
            if pval < pvalue_thres:
                S.add(var)
        
        # Фаза Shrink
        for var in list(S):
            S_temp = S - {var}
            pval = self._conditional_independence_test(target_var, var, S_temp)
            if pval >= pvalue_thres:
                S.remove(var)
        
        return list(S)
    
    def _conditional_independence_test(self, target_var: str, test_var: str, condition_set: set) -> float:
        """
        Выполнение теста на условную независимость.
        
        :param target_var: Имя целевой переменной.
        :param test_var: Имя переменной для проверки.
        :param condition_set: Набор переменных для условия.
        
        :return: p-value, показывающее уровень значимости зависимости.
        """
        x = self.data[target_var].values.reshape(-1, 1)
        y = self.data[test_var].values.reshape(-1, 1)
        z = self.data[list(condition_set)].values if condition_set else None
        
        _, pval = self.CI_test.run_test(x, y, z)
        return pval

