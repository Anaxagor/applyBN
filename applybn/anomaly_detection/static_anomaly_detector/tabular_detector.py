import pandas as pd

from applybn.core.estimators import BNEstimator
from bamt.networks import DiscreteBN, HybridBN
from bamt.nodes.discrete_node import DiscreteNode

from typing import Literal


class TabularDetector:
    def __init__(self,
                 estimator: BNEstimator,
                 has_logit: bool = True,
                 use_mixture: bool = False,
                 has_continuous_data: bool = True,
                 target_name: str = "anomaly"):
        self.estimator = estimator
        self.bn = HybridBN(has_logit, use_mixture) if has_continuous_data else DiscreteBN()
        self.target_name = target_name

    def _inject_target(self, y, data):
        if not self.estimator.bn.edges:
            # todo
            raise Exception

        normal_structure = self.estimator.bn.edges
        # todo: not work for Hybrid: can be signs
        info = self.bn.descriptor
        nodes = self.bn.nodes
        bl_add = [(self.target_name, node_name) for node_name in self.bn.nodes_names]
        nodes += [DiscreteNode(self.target_name)]

        info["types"] |= {self.target_name: "disc"}
        # info["signs"] = {".0": "mimic value to bypass broken check in bamt"}

        self.bn.add_nodes(descriptor=info)

        data[self.target_name] = y

        self.bn.add_edges(data=data,
                          params={"init_edges": list(map(tuple, normal_structure)),
                                  "bl_add": bl_add,
                                  "remove_init_edges": False}
                          )

    def fit(self, discretized_data, y,
            clean_data: pd.DataFrame, descriptor,
            how: Literal["simple", "inject"] = "simple"):
        """
        # todo: doctest format
        Args:
            discretized_data:
            y: pass only if how="inject"
            clean_data:
            descriptor:
            how:

        Returns:

        """
        if how == "inject" and y is None:
            # todo
            raise Exception("no y")
        self.estimator.fit(discretized_data, clean_data=clean_data, descriptor=descriptor, partial=True)

        self.bn = self.estimator.bn
        match how:
            case "simple":
                data_to_parameters_learning = clean_data.copy()
            case "inject":
                self._inject_target(y, discretized_data)
                data_to_parameters_learning = clean_data.copy()
                data_to_parameters_learning[self.target_name] = y
            case _:
                # todo
                raise Exception("Unknown method!")

        substructure = self.bn.find_family(self.target_name, depth=10, height=10)
        print(substructure)
        if len(substructure["edges"]) > 15:
            # todo: warning
            print("Too dense substructure related to target! ", len(substructure["edges"]))

        self.bn.set_structure(
            nodes=[self.bn[node] for node in substructure["nodes"]],
            edges=substructure["edges"],
            info={"types":
                      {name: value for name, value in descriptor["types"].items() if name in substructure["nodes"]}}
        )

        self.bn.fit_parameters(data_to_parameters_learning[substructure["nodes"]])
        return self

    def predict_anomaly(self, X):
        return self.estimator.predict_proba(X)
