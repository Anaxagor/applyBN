import json
import scipy
from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore, IsolationForestScore
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.mixed import ODBPScore

import numpy as np

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, permutation_test_score

from sklearn.metrics import f1_score, make_scorer
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# np.random.seed(20)
# Ecoli dataset
from ucimlrepo import fetch_ucirepo
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


def body(df, y, scorer, cv=20, verbose=False):
    splitter = StratifiedShuffleSplit(n_splits=cv)
    cross_val_scores = []

    estimator = BNEstimator(has_logit=True,
                            use_mixture=False,
                            bn_type="cont")

    discretizer = pp.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

    # create a preprocessor object with encoder and discretizer
    p = Preprocessor([('discretizer', discretizer)])
    # discretize data for structure learning
    discretized_data, encoding = p.apply(df)

    #  get information about data
    info = p.info

    estimator.fit(discretized_data, clean_data=df, partial=True, descriptor=info, progress_bar=False)

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

        # detector.fit(discretized_data, y=None,
        #              clean_data=X_train, descriptor=info,
        #              inject=False, bn_params={"scoring_function": ("K2",),
        #                                       "progress_bar": False})

        detector.partial_fit(X_train, mode="parameters")

        # detector.estimator.bn.get_info(as_df=False)

        outlier_scores = detector.predict(X_test, return_scores=True)

        best_score = my_score(y_test, outlier_scores)

        detector.estimator.bn.distribution = {}

        cross_val_scores.append(best_score)
    return np.mean(cross_val_scores).round(5), np.std(cross_val_scores).round(5)


# ecoli = fetch_ucirepo(id=39)
# df = ecoli.data.features
# y = ecoli.data.targets
# # Among the 8 classes omL, imL, and imS are the minority classes and used as outliers
# y = pd.DataFrame(np.where(np.isin(y, ["omL", "imL", "imS"]), 1, 0))
# print(df.shape)
mat = scipy.io.loadmat('../../../../data/cardio.mat')
df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
df.columns = [f"feature_{i}" for i in range(df.shape[1])]
print(df.shape)

estimator = BNEstimator(has_logit=True,
                        use_mixture=False,
                        bn_type="cont")

model_based_score = ModelBasedScore(estimator)
lof = LocalOutlierScore()
isolation_forest = IsolationForestScore()


# mixed = ODBPScore()

def get_cardio():
    mat = scipy.io.loadmat('../../../../data/cardio.mat')
    df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    return df, y


def get_ecoli():
    ecoli = fetch_ucirepo(id=39)
    df = ecoli.data.features
    y = ecoli.data.targets
    # Among the 8 classes omL, imL, and imS are the minority classes and used as outliers
    y = pd.DataFrame(np.where(np.isin(y, ["omL", "imL", "imS"]), 1, 0))
    return df, y


def get_wbc():
    mat = scipy.io.loadmat('../../../../data/wbc.mat')
    df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    return df, y


conditions = {"score": [model_based_score, lof, isolation_forest],
              "datasets": [get_wbc, get_ecoli, get_cardio]}

mean, std = body(df, y, lof, cv=10)

# d = "ecoli/sss"

# with open(f"{d}/cross_val_cv{cv}.json", "w+") as f:
#     json.dump({"scores": list(cross_val_scores)}, f)

# with open(f"{d}/permutatuion_test_cv{cv}.json", "w+") as f:
#     json.dump({"score": score, "perm_scores": list(permutation_scores), "pvalue": pvalue}, f)
