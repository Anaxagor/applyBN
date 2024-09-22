from typing import Union, List
import numpy as np
import pandas as pd
import time  
from sklearn.feature_selection import mutual_info_classif  
from applybn.core import copy_data
from feature_selector import FeatureSelector


class CausalFeatureSelector(FeatureSelector):
    """
    Класс для отбора признаков на основе причинного эффекта
    """

    def __init__(self, **parameters):
        super().__init__(**parameters)

    @copy_data
    def select_features(
        self, data: Union[pd.DataFrame, np.ndarray], target: Union[str, int]
    ) -> List[Union[str, int]]:
        """
        :param data: Набор данных, содержащий признаки и целевую переменную (если DataFrame, target удаляется из data)
        :param target: Целевая переменная (название столбца для DataFrame или индекс для ndarray)
        :return: Список отобранных признаков и время, затраченное на выполнение отбора.
        """
       
        start_time = time.time()

      
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            X = data.drop(columns=[target]).values
            y = data[target].values
        else:
            feature_names = [f'Feature_{i}' for i in range(data.shape[1])]
            X = data
            y = target

        
        X_discretized = self.discretize_data(X)
        
        
        mutual_info = mutual_info_classif(X_discretized, y, discrete_features=True)
        
        
        sorted_indices = np.argsort(mutual_info)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        
        selected_features = [sorted_features[0]]
        S = [sorted_indices[0]]
        H_Y = self.conditional_entropy(X_discretized[:, [S[0]]], y)
        
        
        for i in range(1, X.shape[1]):
            current_idx = sorted_indices[i]
            current_features = np.column_stack([X_discretized[:, S], X_discretized[:, current_idx]])
            
          
            H_Y_current = self.conditional_entropy(current_features, y)
            CE_Xi_Y = H_Y - H_Y_current

            if CE_Xi_Y != 0:
                S.append(current_idx)
                selected_features.append(sorted_features[i])
                H_Y = H_Y_current
        
      
        end_time = time.time()
        elapsed_time = end_time - start_time
        
     
        return selected_features, elapsed_time

    def discretize_data(self, X_train: np.ndarray) -> np.ndarray:
        """Функция дискретизации данных"""
        R = np.max(X_train) - np.min(X_train)
        IQR = np.percentile(X_train, 75) - np.percentile(X_train, 25)
        n = len(X_train)

        if R == 0 or IQR == 0:
            nh = 100 
        else:
            nh = max((R / (2 * IQR)) * (n ** (1 / 3)), np.log2(n) + 1)
            nh = int(np.ceil(nh))

        bins = np.histogram_bin_edges(X_train, bins=nh)
        X_discretized = np.digitize(X_train, bins) - 1  
        return X_discretized

    def conditional_entropy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Вычисление условной энтропии H(Y|X)"""
        data = pd.DataFrame(X)
        data['Y'] = y
        joint_probs = self.joint_probability(X, y)
        conditional_probs = joint_probs.div(joint_probs.sum(axis=1), axis=0)
        cond_entropy = 0.0
        for x_comb in conditional_probs.index:
            probs = conditional_probs.loc[x_comb].values
            cond_entropy -= np.nansum(probs * np.log(probs + 1e-10))
        return cond_entropy

    def joint_probability(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Вычисление совместной вероятности P(Y, X1, X2, ..., Xk)"""
        data = pd.DataFrame(X)
        data['Y'] = y
        joint_counts = data.groupby(data.columns.tolist()).size().unstack(fill_value=0)
        joint_probs = joint_counts.div(joint_counts.sum().sum())
        return joint_probs

