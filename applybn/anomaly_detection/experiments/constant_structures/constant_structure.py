import json

from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
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
import scipy

def my_score(y, y_pred):
    thresholds = np.linspace(1, y_pred.max(), 100)
    eval_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(y_pred < t, 0, 1)
        eval_scores.append(f1_score(y, outlier_scores_thresholded))

    # plt.figure()
    # ax = sns.lineplot(x=thresholds, y=eval_scores)
    # ax.set(xlabel='thresholds', ylabel='f1_score', title=f"sensitivity analysis")
    # plt.show()
    return np.max(eval_scores)


# ecoli = fetch_ucirepo(id=39)
# df = ecoli.data.features
# y = ecoli.data.targets
# # Among the 8 classes omL, imL, and imS are the minority classes and used as outliers
# y = pd.DataFrame(np.where(np.isin(y, ["omL", "imL", "imS"]), 1, 0))
# print(df.shape)

mat = scipy.io.loadmat('../../../../data/tabular_datasets/cardio.mat')
df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])
df.columns = [f"feature_{i}" for i in range(df.shape[1])]
print(df.shape)
# vehicle datset
# df = pd.read_csv("../../data/tabular_datasets/vehicle_claims_labeled.csv").drop(
#     ['category_anomaly', 'issue_id','breakdown_date', 'repair_date', " Genmodel_ID"], axis=1)

# X_train = df.drop(["class"], axis=1)
# y_train = df["class"]
# X_train, _, y_train, _ = train_test_split(
#     df.drop(["class"], axis=1), df["class"], test_size=0.7, random_state=42, stratify=df["class"])
# print(X_train.shape)
# print(np.unique(y_train))

# Seismic dataset
# data = arff.loadarff('../../data/tabular_datasets/seismic-bumps.arff')
# df = pd.DataFrame(data[0])
# print(df.columns)
# bytes_coded = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
# for col in bytes_coded:
#     df[col] = df[col].str.decode("utf-8")

estimator = BNEstimator(has_logit=True,
                        use_mixture=False,
                        bn_type="cont")

# encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('discretizer', discretizer)])
# discretize data for structure learning
discretized_data, encoding = p.apply(df)

#  get information about data
info = p.info
PROX_STEPS = 45

score_proximity = LocalOutlierScore()

# estimator.fit(X=discretized_data, descriptor=info, partial=True)
# estimator.bn.get_info(as_df=False)
# estimator.bn.save("cardio_entire_data_structure")
#
# print("____")
# estimator.bn.load("cardio_entire_data_structure.json")
# estimator.bn.get_info(as_df=False)
# raise Exception

score = ODBPScore(score_proximity, encoding=encoding, proximity_steps=PROX_STEPS)
local_detector = TabularDetector(estimator,
                                 score=score,
                                 target_name=None)
cv = 10
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=20)
# sss = StratifiedShuffleSplit(n_splits=cv, random_state=20)

cross_val_scores = cross_val_score(local_detector, df, y, scoring=make_scorer(my_score),
                         cv=skf, verbose=2,
                         params={"structure": "cardio_entire_data_structure.json"}, error_score="raise")
print(cross_val_scores)
print(cross_val_scores.mean().round(5),
      cross_val_scores.std().round(5))

score, permutation_scores, pvalue = permutation_test_score(local_detector, df, y, scoring=make_scorer(my_score),
                                cv=skf, verbose=2, n_jobs=10, n_permutations=1000,
                                fit_params={"structure": "cardio_entire_data_structure.json"}, random_state=20)
print()
print(f"Original Score: {score:.3f}")
print(permutation_scores.mean().round(5), permutation_scores.std().round(5))
print(f"P-value: {pvalue:.3f}")


# random_state=20
# 5
# 0.86667 0.1633
# 10
# 0.72163 0.42662
# 15
# 0.41818 0.43788
# 20
# 0.21991 0.31622

# i = -1
# cross_val_scores = {"scores": []}
# for train_indexes, test_indexes in skf.split(df, y):
#     i += 1
#     print(i)
#     local_detector = TabularDetector(estimator,
#                                      score=score,
#                                      target_name=None)
#
#     X_train, X_test = df.iloc[train_indexes, :], df.iloc[test_indexes, :]
#     y_train, y_test = y.iloc[train_indexes, :], y.iloc[test_indexes, :]
#
#     # local_detector.estimator.bn.fit_parameters(X_train)
#     local_detector.fit(X_train, structure="ecoli_entire_data.json")
#     outlier_scores = local_detector.predict(X_test)
#
#     final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1), y_test.values.reshape(-1, 1).astype(int)]),
#                          columns=["score", "anomaly"])
#
#     thresholds = np.linspace(1, outlier_scores.max(), 100)
#     eval_scores = []
#
#     for t in thresholds:
#         outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
#         eval_scores.append(f1_score(y_test.values, outlier_scores_thresholded))
#
#     # plt.figure(figsize=(20, 12))
#     # desc = f"""[Nonlinear, no scaler, model metric:Z-score, prox_step: {PROX_STEPS}]"""
#     # sns.scatterplot(data=final, x=range(final.shape[0]), s=20,
#     #                 y="score", hue="anomaly") \
#     #     .set_title("Scores; Impacts(P, M): "
#     #                   f"[{detector.score.proximity_impact.round(3)}, {detector.score.model_impact.round(3)}]")
#
#     # plt.figure()
#     # ax = sns.lineplot(x=thresholds, y=eval_scores)
#     # ax.set(xlabel='thresholds', ylabel='f1_score', title=f"{i}: sensitivity analysis")
#     # plt.show()
#
#     cross_val_scores["scores"].append(np.max(eval_scores))
#
# print(f"Mean: {np.mean(cross_val_scores['scores']).round(5)}\n"
#       f"Std: {np.std(cross_val_scores['scores']).round(5)}")

# 5
# Mean: 0.85333
# Std: 0.12927
# 10
# Mean: 0.83333
# Std: 0.30732
# 15
d = "cardio/skf"

with open(f"{d}/cross_val_cv{cv}.json", "w+") as f:
    json.dump({"scores": list(cross_val_scores)}, f)

with open(f"{d}/permutatuion_test_cv{cv}.json", "w+") as f:
    json.dump({"score": score, "perm_scores": list(permutation_scores), "pvalue": pvalue}, f)
