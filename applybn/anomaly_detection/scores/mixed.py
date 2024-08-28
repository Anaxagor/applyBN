import pandas as pd

from applybn.anomaly_detection.scores.score import Score
# from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


class ODBPScore(Score):
    def __init__(self, BNEstimator, score, encoding, proximity_steps=10):
        super().__init__()
        self.estimator = BNEstimator
        self.score_proximity = score
        self.encoding = encoding
        self.proximity_steps = proximity_steps

        self.proximity_impact = 0
        self.model_impact = 0

    def local_model_score(self, X: pd.DataFrame, node_name):
        node = self.estimator.bn[node_name]
        diff = []
        dist = self.estimator.bn.distributions[node_name]
        parents = node.cont_parents + node.disc_parents

        for _, row in X.iterrows():
            # todo: disgusting
            pvalues = row[parents].to_dict()

            pvals_bamt_style = [pvalues[parent] for parent in parents]
            cond_dist = self.estimator.bn.get_dist(node_name, pvals=pvalues)

            # todo: super disgusting
            if isinstance(cond_dist, tuple):
                cond_mean = cond_dist[0]
            else:
                dispvals = []
                for pval in pvals_bamt_style:
                    if isinstance(pval, str):
                        dispvals.append(pval)

                if "vals" in dist.keys():
                    classes = dist["vals"]
                elif "classes" in dist.keys():
                    classes = dist["classes"]
                elif "hybcprob" in dist.keys():
                    if "classes" in dist["hybcprob"][str(dispvals)]:
                        classes = dist["hybcprob"][str(dispvals)]["classes"]
                    else:
                        raise Exception()
                else:
                    raise Exception()

                classes_coded = np.asarray([self.encoding[node_name][class_name] for class_name in classes])
                cond_mean = classes_coded @ np.asarray(cond_dist).T
            if isinstance(row[node_name], str):
                # MAE
                diff.append(abs(cond_mean - self.encoding[node_name][row[node_name]]))
            else:
                # MAE
                diff.append(abs(cond_mean - row[node_name]))

        # scaler = StandardScaler()
        # scaler = MinMaxScaler()
        # diff_scaled = scaler.fit_transform(np.asarray(diff).reshape(-1, 1))

        # return diff_scaled
        return np.asarray(diff).reshape(-1, 1)

    def local_proximity_score(self, X):
        t = np.random.randint(X.shape[1] // 2, X.shape[1] - 1)
        columns = np.random.choice(X.columns, t, replace=False)

        subset = X[columns]

        subset_cont = subset.select_dtypes(include=["number"])
        outlier_factors = self.score_proximity.score(subset_cont)

        # scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(0, 10))
        # outlier_factors_scaled = scaler.fit_transform(outlier_factors.reshape(-1, 1))

        return np.asarray(outlier_factors).reshape(-1, 1)
        # return outlier_factors_scaled

    def score(self, X):
        child_nodes = []
        for column in X.columns:
            if self.estimator.bn[column].disc_parents + self.estimator.bn[column].cont_parents:
                child_nodes.append(column)

        proximity_factors = []
        model_factors = []

        for _ in range(self.proximity_steps):
            proximity_factors.append(self.local_proximity_score(X))

        for child_node in child_nodes:
            model_factors.append(self.local_model_score(X, child_node))

        proximity_factors = np.hstack(proximity_factors)
        model_factors = np.hstack(model_factors)

        proximity_factors = np.where(proximity_factors <= 0, proximity_factors, 0)
        model_factors = np.where(model_factors >= 0, model_factors, 0)

        proximity_outliers_factors = np.negative(proximity_factors).sum(axis=1)
        model_outliers_factors = model_factors.sum(axis=1)

        # outlier_factors = proximity_outliers_factors + model_outliers_factors

        outlier_factors = np.vstack([proximity_outliers_factors, model_outliers_factors])
        from_proximity = np.where(outlier_factors[:, 0] > outlier_factors[:, 1], 1, 0)

        test = np.max(np.vstack([proximity_outliers_factors, model_outliers_factors]), axis=0)
        # model_impact = model_outliers_factors / outlier_factors
        # proximity_impact = proximity_outliers_factors / outlier_factors
        #
        # self.model_impact = np.nanmean(model_impact)
        # self.proximity_impact = np.nanmean(proximity_impact)
        # fig, ax = plt.subplots()
        #
        # # Stacked bar chart
        # ax.bar(range(outlier_factors.shape[0]), model_impact, label="model_impact")
        # ax.bar(range(outlier_factors.shape[0]), proximity_impact, bottom=model_impact,
        #        label="prox_impact")
        # ax.legend()
        # ax.set_title(
        #     f"Mean impact model: {self.model_impact.round(3)}; Mean impact proximity: {self.proximity_impact.round(3)}")
        # plt.show()

        # return outlier_factors
        return test, from_proximity
