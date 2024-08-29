from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore

import matplotlib.pyplot as plt
import numpy as np

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.io import arff
import scipy
import pandas as pd

# data = arff.loadarff('../../data/tabular_datasets/seismic-bumps.arff')
# df = pd.DataFrame(data[0])
# print(df.columns)
# bytes_coded = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
# for col in bytes_coded:
#     df[col] = df[col].str.decode("utf-8")

mat = scipy.io.loadmat('../../data/tabular_datasets/wbc.mat')
df, y = pd.DataFrame(mat["X"]), pd.DataFrame(mat["y"])

# df["anomaly"] = y
# sns.pairplot(data=df, hue='anomaly', corner=True)
# plt.tight_layout()
# plt.savefig("tmp.png")
# df = pd.read_csv("../../data/tabular_datasets/vehicle_claims_labeled.csv").drop(
#     ['category_anomaly', 'issue_id','breakdown_date', 'repair_date', " Genmodel_ID"], axis=1)

# X_train = df.drop(["class"], axis=1)
# y_train = df["class"]
# X_train, _, y_train, _ = train_test_split(
#     df.drop(["class"], axis=1), df["class"], test_size=0.7, random_state=42, stratify=df["class"])
# print(X_train.shape)
# print(np.unique(y_train))
estimator = BNEstimator(has_logit=True,
                        bn_type="cont")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
# p = Preprocessor([('discretizer', discretizer)])
# discretize data for structure learning
discretized_data, encoding = p.apply(df)

# for k, v in encoding.items():
#     discretized_data[k] += 1
#     for k1 in v.keys():
#         encoding[k][k1] += 1

# y_coder = pp.LabelEncoder()
# disc_y = pd.Series(y_coder.fit_transform(y_train),
#                    name=y_train.name,
#                    index=y_train.index)
# print(dict(zip(y_coder.classes_, range(len(y_coder.classes_)))))

# # get information about data
info = p.info

# # ------------------

# score = ModelBasedScore(estimator)
score_proximity = LocalOutlierScore()
score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=100)

detector = TabularDetector(estimator,
                           score=score,
                           target_name=None)

detector.fit(discretized_data, y=None,
             clean_data=df, descriptor=info,
             inject=False, bn_params={"scoring_function": ("MI",)})
detector.estimator.bn.get_info(as_df=False)

outlier_scores = detector.detect(df, return_scores=True)

final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1), y.values.reshape(-1, 1).astype(int)]),
                     columns=["score", "anomaly"])

thresholds = np.linspace(1, outlier_scores.max(), 100)
eval_scores = []

for t in thresholds:
    outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
    eval_scores.append(recall_score(y.values, outlier_scores_thresholded))

# plt.figure()
# sns.scatterplot(data=final, x=range(final.shape[0]), y="score", hue="anomaly")
# plt.show()

plt.figure()
ax = sns.lineplot(x=thresholds, y=eval_scores)
ax.set(xlabel='thresholds', ylabel='recall', title="sensitivity analysis")
plt.show()
