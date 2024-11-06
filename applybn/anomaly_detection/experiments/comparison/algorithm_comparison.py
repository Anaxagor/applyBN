# import json
import scipy
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore, IsolationForestScore
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.mixed import ODBPScore

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import f1_score, make_scorer

# np.random.seed(20)
from bamt.log import bamt_logger
from tqdm import tqdm

bamt_logger.switch_console_out(False)


def my_score(y, y_pred):
    thresholds = np.linspace(1, y_pred.max(), 100)
    eval_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(y_pred < t, 0, 1)
        eval_scores.append(f1_score(y, outlier_scores_thresholded))

    return np.max(eval_scores)


def body(df_getter, scorer_class, cv=10, verbose=False, additional_scorer=None):
    splitter = StratifiedShuffleSplit(n_splits=cv)

    match scorer_class.__name__:
        case "ModelBasedScore":
            df, y, estimator = learn_structure(df_getter, full=True)
            if not estimator:
                return np.nan, np.nan
            scorer = scorer_class(estimator)
        case "LocalOutlierScore" | "IsolationForestScore":
            df, y, estimator = learn_structure(df_getter)
            scorer = scorer_class()
        case "ODBPScore":
            df, y, estimator, encoding = learn_structure(df_getter, return_encoding=True)
            if not additional_scorer:
                raise Exception("Need prox scorer (additional scorer param)!!")
            scorer = scorer_class(additional_scorer, encoding)
        case _:
            raise Exception("Unknown score model!")
    print("Structure -> FINISHED!")
    cross_val_scores = []

    if verbose:
        iterator = tqdm(splitter.split(df, y), total=splitter.get_n_splits(), )
    else:
        iterator = splitter.split(df, y)

    for train_indexes, test_indexes in iterator:
        X_train, X_test = df.iloc[train_indexes, :], df.iloc[test_indexes, :]
        y_train, y_test = y.iloc[train_indexes, :], y.iloc[test_indexes, :]

        detector = TabularDetector(estimator,
                                   score=scorer,
                                   target_name=None)

        # detector.estimator.bn.get_info(as_df=False)

        detector.partial_fit(X_train, mode="parameters")

        outlier_scores = detector.predict(X_test, return_scores=True)
        best_score = my_score(y_test, outlier_scores)
        # final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1), y_test.values.reshape(-1, 1).astype(int)]),
        #                  columns=["score", "anomaly"])
        # plt.figure(figsize=(20, 12))
        # sns.scatterplot(data=final, x=range(final.shape[0]), s=20,
        #                 y="score", hue="anomaly")
        #
        # plt.show()
        detector.estimator.bn.distribution = {}

        cross_val_scores.append(best_score)
    return np.mean(cross_val_scores).round(5), np.std(cross_val_scores).round(5)


def get_cardio():
    # mat = scipy.io.loadmat('../data/tabular/cardio.mat')
    # df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
    # df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    df = pd.read_csv("../data/tabular/cardio.csv", index_col=0)
    return df, pd.DataFrame(df.pop("anomaly"))


def get_ecoli():
    df = pd.read_csv("../data/tabular/ecoli.csv")
    return df, pd.DataFrame(df.pop("y"))


def get_wbc():
    # mat = scipy.io.loadmat('../data/tabular/wbc.mat')
    # df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
    # df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    df = pd.read_csv("../data/tabular/wbc.csv", index_col=0)
    return df.iloc[:, :20], pd.DataFrame(df.pop("anomaly"))


def learn_structure(df_getter: callable, full=False, return_encoding=False):
    df, y = df_getter()
    if full:
        df['y'] = y.iloc[:, 0].to_numpy().astype(int)
        bn_type = "hybrid"
    else:
        bn_type = "cont"

    estimator = BNEstimator(has_logit=True,
                            use_mixture=False,
                            bn_type=bn_type)

    discretizer = pp.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

    # create a preprocessor object with encoder and discretizer
    p = Preprocessor([('discretizer', discretizer)])
    # discretize data for structure learning
    discretized_data, encoding = p.apply(df)

    #  get information about data
    info = p.info
    if full:
        bl_add = [('y', node_name) for node_name in df.columns]
        bn_params = {"params": {"bl_add": bl_add}}
    else:
        bn_params = {}

    estimator.fit(discretized_data, clean_data=df, partial=True,
                  descriptor=info, **bn_params) # , progress_bar=False

    if full:
        if not estimator.bn["y"].disc_parents + estimator.bn["y"].cont_parents:
            return df, y, None

    if return_encoding:
        return df, y, estimator, encoding
    else:
        return df, y, estimator


model_based_score = ModelBasedScore
lof = LocalOutlierScore
isolation_forest = IsolationForestScore
mixed_score = ODBPScore


# conditions = {"score": [model_based_score, lof, isolation_forest],
#               "df_getter_args": [get_wbc]}

conditions = {"score": [mixed_score],
              "df_getter_args": [get_wbc]}

grid = ParameterGrid(conditions)

df, y = get_ecoli()
print(df)
# df["anomaly"] = y.values.astype(int)
# df.to_csv("../data/tabular/wbc.csv", index=False)
# for params in grid:
#     print(params)
#     print(body(params["df_getter_args"], scorer_class=params["score"], additional_scorer=LocalOutlierScore()))
