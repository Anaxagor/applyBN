import pandas as pd

from applybn.anomaly_detection.scores.score import Score
# from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerPathCollection
from sklearn.decomposition import PCA


class ODBPScore(Score):
    def __init__(self, BNEstimator, score, encoding, proximity_steps=10):
        super().__init__()
        self.estimator = BNEstimator
        self.score_proximity = score
        self.encoding = encoding
        self.proximity_steps = proximity_steps

        self.proximity_impact = 0
        self.model_impact = 0


    def local_model_score_linear(self, X: pd.DataFrame, node_name):
        node = self.estimator.bn[node_name]
        parents = node.cont_parents + node.disc_parents
        subspace = X[[node_name] + parents]
        means = []
        dist = self.estimator.bn.distributions[node_name]

        # if node.disc_parents:
        #     grouped = subspace.groupby(node.disc_parents)
        # else:
        #     grouped = subspace

        for indx, row in subspace.iterrows():
            coefs = dist["regressor_obj"].coef_
            mean_estimated = dist["regressor_obj"].intercept_ + coefs.reshape(1, -1) @ row[1:].to_numpy().reshape(-1, 1)
            true_value = row[node_name]
            # Z-score
            means.append((true_value - mean_estimated[0][0]) / dist["variance"])

            # MAE
            # means.append(abs(true_value - mean_estimated[0][0]))

        # scaler = StandardScaler()
        # mean_scaled = scaler.fit_transform(np.asarray(means).reshape(-1, 1))
        return np.asarray(means).reshape(-1, 1)
        # return mean_scaled

        # for parents_combination, group in grouped:
        #     diff_local = []
        #     mean_node = dist[parents_combination].regressor_obj.coef_
        #     subspace = [mean_node]
        #     subspace.extend(
        #         [self.estimator.bn.get_dist(parent_name) for parent_name in group.columns]
        #     )
        #     subspace = np.array(subspace)
        #     # todo: matrix form needed
        #     for indx, row in group.iterrows():
        #         diff_local.append(mean_node + subspace @ row.to_numpy())
        #
        #     diff.append(diff_local)

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
                if len(cond_dist) == 2:
                    cond_mean, var = cond_dist
                else:
                    cond_mean = node.predict(dist, pvals=pvals_bamt_style)
                    # todo: may be use singular vals of cov matrix as norm constants?
                    diff.append(row[node_name] - cond_mean)
                    continue
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
                        if pd.isna(classes[0]):
                            # if subspace of a combination is empty
                            diff.append(np.nan)
                            continue
                    else:
                        raise Exception()
                else:
                    raise Exception()

                classes_coded = np.asarray([self.encoding[node_name][class_name] for class_name in classes])
                cond_mean = classes_coded @ np.asarray(cond_dist).T
            if isinstance(row[node_name], str):
                # MAE
                # diff.append(abs(cond_mean - self.encoding[node_name][row[node_name]]))
                diff.append(
                    self.encoding[node_name][row[node_name]] - cond_mean
                )
            else:
                # MAE
                # diff.append(abs(cond_mean - row[node_name]))

                # Z score
                diff.append(
                    (row[node_name] - cond_mean) / var
                )

        # scaler = StandardScaler()
        # scaler = MinMaxScaler()
        # diff_scaled = scaler.fit_transform(np.asarray(diff).reshape(-1, 1))

        # return diff_scaled
        return np.asarray(diff).reshape(-1, 1)

    @staticmethod
    def plot_lof(X, negative_factors):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        def update_legend_marker_size(handle, orig):
            handle.update_from(orig)
            handle.set_sizes([20])

        if X.shape[1] > 2:
            pca = PCA(n_components=3)
            X = pca.fit_transform(X)
            print(pca.explained_variance_ratio_)

        plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
        # plot circles with radius proportional to the outlier scores
        radius = (negative_factors.max() - negative_factors) / (negative_factors.max() - negative_factors.min())
        scatter = plt.scatter(
            X[:, 0],
            X[:, 1],
            s=1000 * radius,
            edgecolors="r",
            facecolors="none",
            label="Outlier scores",
        )
        plt.axis("tight")
        plt.legend(
            handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
        )
        plt.title("Local Outlier Factor (LOF)")
        plt.show()

    def local_proximity_score(self, X):
        t = np.random.randint(X.shape[1] // 2, X.shape[1] - 1)
        columns = np.random.choice(X.columns, t, replace=False)

        subset = X[columns]

        subset_cont = subset.select_dtypes(include=["number"])

        # The higher, the more abnormal
        outlier_factors = self.score_proximity.score(subset_cont)
        # plt.hist(outlier_factors)
        # plt.show()
        # self.plot_lof(subset_cont, outlier_factors)

        # scaler = StandardScaler()
        # scaler = MinMaxScaler(feature_range=(0, 10))
        # outlier_factors_scaled = scaler.fit_transform(outlier_factors.reshape(-1, 1))
        # self.plot_lof(subset_cont, outlier_factors_scaled)
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

        # make zero impact from factors less than 0 since they correspond to inliners
        proximity_factors = np.where(proximity_factors <= 0, 0, proximity_factors)

        # higher the more normal, only
        proximity_outliers_factors = proximity_factors.sum(axis=1)

        # any sign can be here, so we take absolute values since distortion from mean is treated as anomaly
        model_outliers_factors = np.abs(model_factors).sum(axis=1)

        outlier_factors = proximity_outliers_factors + model_outliers_factors

        model_impact = model_outliers_factors / outlier_factors
        proximity_impact = proximity_outliers_factors / outlier_factors

        self.model_impact = np.nanmean(model_impact)
        self.proximity_impact = np.nanmean(proximity_impact)

        return outlier_factors
