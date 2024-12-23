from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class BNFeatureGenerator(BaseEstimator, TransformerMixin):
    """Bayesian Network feature generator."""
    # TODO: Implement Bayesian Network feature generator, use TransformerMixin as reference
    def __init__(self, *, param=1):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Generate new features
        return X
