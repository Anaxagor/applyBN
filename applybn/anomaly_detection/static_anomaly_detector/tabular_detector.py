import pandas as pd

from applybn.core.estimators import BNEstimator

# todo: aliases??
# from typing import Type
from applybn.core.schema import scores
import inspect


class TabularDetector:
    def __init__(self,
                 estimator: BNEstimator,
                 score: scores,
                 target_name: str | None = "anomaly"):
        self.estimator = estimator
        self.score = score
        self.scores_ = None
        self.target_name = target_name

    def partial_fit(self, X, mode: str, y=None, **kwargs):
        match mode:
            case "structure":
                self.estimator.fit(X, partial=True, **kwargs)
            case "parameters":
                self.estimator.bn.fit_parameters(X, **kwargs)
                self.score.bn = self.estimator.bn
            case "both":
                self.estimator.fit(X, **kwargs)
            case _:
                raise Exception("Unknown mode!")

    def fit(self, discretized_data, y, descriptor,
            clean_data: pd.DataFrame | None = None,
            inject=False, bn_params=None):
        """
        # todo: doctest format
        Args:
            discretized_data:
            y: pass only if how="inject"
            clean_data:
            descriptor:
            inject:
            bn_params:
        Returns:

        """
        if bn_params is None:
            bn_params = {}
        if inject and y is None:
            # todo
            raise Exception("no y")

        self.estimator.fit(discretized_data, clean_data=clean_data,
                           descriptor=descriptor, partial=True, **bn_params)
        data_to_parameters_learning = clean_data.copy()
        if inject:
            self.estimator.inject_target(y=y, data=discretized_data)
            data_to_parameters_learning = clean_data.copy()
            data_to_parameters_learning[self.target_name] = y.to_numpy()

        if not self.target_name:
            self.estimator.bn.fit_parameters(data_to_parameters_learning)
            return self

        if self.target_name not in discretized_data:
            # todo
            raise Exception("column with target name was not found.")

        substructure = self.estimator.bn.find_family(self.target_name, depth=10, height=10)
        if len(substructure["edges"]) > 15:
            # todo: warning
            print("Too dense substructure related to target! ", len(substructure["edges"]))

        self.estimator.bn.set_structure(
            nodes=[self.estimator.bn[node] for node in substructure["nodes"]],
            edges=substructure["edges"],
            info={"types":
                      {name: value for name, value in descriptor["types"].items() if name in substructure["nodes"]},
                  "signs": {name: value for name, value in descriptor["signs"].items() if
                            name in substructure["nodes"]}}
        )

        self.estimator.bn.fit_parameters(data_to_parameters_learning[substructure["nodes"]])
        return self

    def predict(self,
                X,
                threshold=0.7,
                return_scores=True,
                inverse_threshold=False):
        # todo: fit validation

        scores_values = self.score.score(X)
        scores_values = pd.Series(scores_values.flatten(), index=X.index)

        if return_scores:
            return scores_values

        # self.scores = scores_values
        # classes = scores_values.copy()
        #
        # if isinstance(threshold, str):
        #     match threshold:
        #         case "mean_score":
        #             threshold = scores_values.mean()
        #             print(f"Threshold: {threshold}")
        #             # todo: INVERSE
        #             classes[scores_values >= threshold] = 1
        #             classes[scores_values < threshold] = 0
        #         case _:
        #             raise Exception()
        # else:
        #     classes[scores_values >= threshold] = 0
        #     classes[scores_values < threshold] = 1

        # return classes

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
