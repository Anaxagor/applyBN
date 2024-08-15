import pandas as pd

from applybn.core.estimators import BNEstimator

# todo: aliases??
# from typing import Type
from applybn.core.schema import scores


class TabularDetector:
    def __init__(self,
                 estimator: BNEstimator,
                 score: scores,
                 target_name: str | None = "anomaly"):
        self.estimator = estimator
        self.score = score
        self.scores = None
        self.target_name = target_name

    def fit(self, discretized_data, y,
            clean_data: pd.DataFrame, descriptor,
            inject=False):
        """
        # todo: doctest format
        Args:
            discretized_data:
            y: pass only if how="inject"
            clean_data:
            descriptor:
            inject:
        Returns:

        """
        if inject and y is None:
            # todo
            raise Exception("no y")

        self.estimator.fit(discretized_data, clean_data=clean_data, descriptor=descriptor, partial=True)
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

    def detect(self,
               X,
               threshold=0.7,
               return_scores=False,
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
