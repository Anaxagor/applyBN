import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from bamt.networks import DiscreteBN, HybridBN, ContinuousBN

from typing import Unpack, Literal
from applybn.core.schema import ParamDict


class BNEstimator(BaseEstimator):
    _parameter_constraints = {
        "has_logit": [bool],
        "use_mixture": [bool],
        "bn_type": [str]
    }

    def __init__(self,
                 has_logit: bool = False,
                 use_mixture: bool = False,
                 bn_type: Literal["hybrid", "disc", "cont"] = "hybrid"
                 ):
        str2net = {"hybrid": HybridBN, "disc": DiscreteBN, "cont": ContinuousBN}
        self.has_logit = has_logit
        self.use_mixture = use_mixture
        self.bn_type = bn_type
        # self.has_continuous_data = has_continuous_data
        # self.target_name = target_name
        # we're assuming here that anomaly node is always disc, thus cont_bn cannot be here
        # todo: hardcode
        match bn_type:
            case "hybrid":
                params = dict(use_mixture=use_mixture, has_logit=has_logit)
            case "cont":
                params = dict(use_mixture=use_mixture)
            case "disc":
                params = dict()
            case _:
                # todo
                raise Exception("Unknown bn type!")
        self.bn = str2net[bn_type](**params)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X,
            # inverse_transformer: callable,
            clean_data,
            descriptor: dict,
            y=None,
            partial: bool = False,
            **kwargs: Unpack[ParamDict]):
        """
        # todo: remove passing 2 dataframes, instead use partial arg
        # todo: doctest format
        Args:
            clean_data: data with no preprocessing
            # inverse_transformer: a method to get original data to learn parameters
            X : data preprocessed, e.g., have no cont data if score func is K2, read more in docs
            y: not used
            descriptor:
            partial: if False - learn structure and parameters, otherwise only structure

        Returns:
            self : object
                Returns self.
        """

        self.bn.add_nodes(descriptor)
        self.bn.add_edges(X, **kwargs)

        if not partial:
            self.bn.fit_parameters(clean_data)

        return self

    def predict_proba(self, X: pd.DataFrame):
        # check_is_fitted(self)
        # X = check_array(X)
        # todo: turn into vectors? very slow
        probas = []
        for indx, row in X.iterrows():
            try:
                result = self.bn.get_dist("y", pvals=row.to_dict())[0]
            except KeyError:
                print("Break")
                result = np.nan
            probas.append(result)

        return pd.Series(probas)
