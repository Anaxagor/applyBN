from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns

df = pd.read_csv("../../data/benchmarks/bank_data/bank_data.csv", index_col=0) \
    .sample(3000, random_state=42, ignore_index=True)

disc_cols = df.select_dtypes(include=["object"]).columns
cont_cols = df.select_dtypes(include=["int64"]).columns
df[cont_cols] = df[cont_cols].astype(float)

# print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["y"], axis=1), df["y"], test_size=0.5, random_state=42, stratify=df["y"])

estimator = BNEstimator(has_logit=True,
                        bn_type="hybrid")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
# discretize data for structure learning

discretized_data, encoding = p.apply(X_train)

for k, v in encoding.items():
    discretized_data[k] += 1
    for k1 in v.keys():
        encoding[k][k1] += 1

y_coder = pp.LabelEncoder()
disc_y = pd.Series(y_coder.fit_transform(y_train),
                   name=y_train.name,
                   index=y_train.index)
print(dict(zip(y_coder.classes_, range(len(y_coder.classes_)))))

# get information about data
info = p.info

# ------------------

# score = ModelBasedScore(estimator)
score_proximity = LocalOutlierScore(n_neighbors=45)
score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=20)

detector = TabularDetector(estimator,
                           score=score,
                           target_name=None)

detector.fit(discretized_data, y=None,
             clean_data=X_train, descriptor=info,
             inject=False)
detector.estimator.bn.get_info(as_df=False)

# discretized_test, _ = p.apply(X_test)
outlier_scores = detector.detect(X_train, return_scores=True)
threshold = 10
outlier_scores[outlier_scores >= threshold] = 1
outlier_scores[outlier_scores < threshold] = 0
# plt.figure(figsize=(20, 5))
# plt.plot(outlier_scores.sort_index(), label="Outlier_score")
#
# plt.scatter(x=disc_y[disc_y == 1].index,
#             y=outlier_scores[disc_y == 1],
#             label="anomalies", color="r", s=4)
#
# plt.title("Preds")
# plt.legend()
# plt.show()
f, ax = plt.subplots(1, 1)

sns.kdeplot(detector.scores, ax=ax).set_title(f"Scores kdeplot")

import matplotlib.patches as mpatches
mim_value = detector.score.model_impact.round(3)
mip_value = detector.score.proximity_impact.round(3)
f1_value = f1_score(disc_y, outlier_scores).round(3)

mim = mpatches.Patch(color='red', label=f"Impact model: {mim_value}")
mip = mpatches.Patch(color='black', label=f"Impact proximity: {mip_value}")
f1 = mpatches.Patch(color="blue", label=f"F1_Score: {f1_value}")

leg = ax.legend(handles=[mim, mip, f1])
# leg.get_frame().set_edgecolor('b')
# leg.get_frame().set_linewidth(0.0)
# sns.move_legend(ax, "center", bbox_to_anchor=(2, 1))
# plt.tight_layout()
plt.savefig(f"tmp.png")
# scores = []
#
# for i in np.linspace(0, 1, 10):
#     preds_trunc = np.where(preds > i, 0, 1)
#     scores.append(f1_score(y_coder.transform(y_test), preds_trunc))

# plt.plot(np.linspace(0, 1, 10), scores)
# plt.xlabel("threshold")
# plt.ylabel("f1_score")
# plt.title("Threshold analysis on bank data (hybrid BN)")
# plt.show()
# plt.savefig("B001_bank_data_inject.png")
# detector.estimator.bn.plot("tmp.html")
# print(
#     detector.bn.find_family("y", height=3, depth=3, plot_to="tt5.html")
# )
