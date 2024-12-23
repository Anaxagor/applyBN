from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
import numpy as np


class BNFeatureSelector(BaseEstimator, SelectorMixin):
    """Bayesian Network feature selector."""

    # TODO: Implement Bayesian Network feature selector, use SelectorMixin as reference
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[:2] = True
        return mask
