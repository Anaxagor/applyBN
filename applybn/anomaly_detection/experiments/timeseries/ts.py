# import json
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
from sklearn.model_selection import ParameterGrid, StratifiedShuffleSplit

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
# np.random.seed(20)
from bamt.log import bamt_logger

# bamt_logger.switch_console_out(False)
from bamt.networks.continuous_bn import ContinuousBN


def get_creditcard():
    df1 = pd.read_csv("../data/ts/creditcard_p1.csv", sep=",", index_col=0).drop(["Amount", "Time"], axis=1)
    df2 = pd.read_csv("../data/ts/creditcard_p2.csv", sep=",", index_col=0).drop(["Amount", "Time"], axis=1)

    df = pd.concat(([df1, df2]))
    y = df.pop("Class")
    return df, pd.DataFrame(y)


def get_scab():
    df = pd.read_csv("../data/ts/SKAB.csv", index_col=0)
    return df, pd.DataFrame(df.pop("anomaly"))


def get_asd():
    # application server dataset
    df = pd.read_csv("../data/ts/InterFusion.csv", index_col=0)
    y = df.pop("anomaly")
    return df, pd.DataFrame(y)


def my_score(y, y_pred):
    thresholds = np.linspace(1, y_pred.max(), 100)
    eval_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(y_pred < t, 0, 1)
        eval_scores.append(f1_score(y, outlier_scores_thresholded))

    return np.max(eval_scores)


def body(df_getter, scorer_class, cv=10, additional_scorer=None):
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
            df, y, estimator, encoding = learn_structure(df_getter,
                                                         return_encoding=True)
            if not additional_scorer:
                raise Exception("Need prox scorer (additional scorer param)!!")
            scorer = scorer_class(additional_scorer, encoding, proximity_steps=5)
        case _:
            raise Exception("Unknown score model!")
    print("Structure -> FINISHED!")
    cross_val_scores = []

    # detector = TabularDetector(estimator,
    #                            score=scorer,
    #                            target_name=None)
    #

    # detector.partial_fit(df, mode="parameters")

    # detector.estimator.bn.save("K2_creditcard_params")
    for train_indexes, test_indexes in splitter.split(df, y):
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
        print(cross_val_scores)
    return np.mean(cross_val_scores).round(5), np.std(cross_val_scores).round(5)


def learn_structure(df_getter: callable, full=False, return_encoding=False):
    df, y = df_getter()

    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

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
    # bn = ContinuousBN(use_mixture=False)
    # bn.load("K2_creditcard.json")
    # estimator.bn = bn

    # return df, y, estimator, encoding

    #  get information about data
    info = p.info
    if full:
        bl_add = [('y', node_name) for node_name in df.columns]
        bn_params = {"params": {"bl_add": bl_add}}
    else:
        bn_params = {}

    estimator.fit(discretized_data, clean_data=df, partial=True,
                  descriptor=info, **bn_params)  # , progress_bar=False

    estimator.bn.get_info(as_df=False)

    # estimator.bn.save("K2_skab")

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
              "df_getter_args": [get_creditcard],}

grid = ParameterGrid(conditions)

for params in grid:
    print(params)
    print(body(params["df_getter_args"], scorer_class=params["score"], additional_scorer=LocalOutlierScore()))

# Creditcard
# K2
# (0.35453, 0.06328)

# SKAB
# K2
# (0.45097, 0.00224)

# Server
# (0.21397, 0.05233)

