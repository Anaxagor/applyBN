from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.model_based import ModelBasedScore
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
import seaborn as sns

np.random.seed(20)

MEAN_F1, VAR_F1 = 4.5, 10
MEAN_F2, VAR_F2 = 3.5, 5


def f(x1, x2, a, b):
    return a*x1 + b*x2 + a * b


def bomb(df):
    shifts = np.random.normal(loc=180, scale=5, size=1000)
    p = 0.01
    indexes_left = np.random.choice([1, 0], 1000, p=[p, 1 - p], )
    shifts[indexes_left == 0] = 0

    df["feature3"] += shifts
    df["anomaly"] = indexes_left
    return df


A = 5
B = 10

df = pd.DataFrame({"feature1": np.random.normal(scale=VAR_F1, size=1000, loc=MEAN_F1),
                   "feature2": np.random.normal(scale=VAR_F2, size=1000, loc=MEAN_F2),
                   })

df["feature3"] = f(df["feature1"], df["feature2"], A, B)
df = bomb(df)
# sns.pairplot(data=df, hue="anomaly")
# plt.tight_layout()
# plt.savefig("tmp.png")

y_train = df.pop("anomaly")

estimator = BNEstimator(has_logit=True,
                        bn_type="cont")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('discretizer', discretizer)])
discretized_data, encoding = p.apply(df)
info = p.info

score_proximity = LocalOutlierScore(n_neighbors=20)
score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=10)

detector = TabularDetector(estimator,
                           score=score,
                           target_name=None)

detector.fit(discretized_data, y=None,
             clean_data=df, descriptor=info,
             inject=False)
detector.estimator.bn.get_info(as_df=False)

outlier_scores, from_prox = detector.detect(df, return_scores=True)

final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1),
                                y_train.values.reshape(-1, 1)]),
                     columns=["scores", "anomaly"])

final["from_prox"] = from_prox

print(
    f"Model impact: {detector.score.model_impact} \n"
    f"Proximity impact: {detector.score.proximity_impact}")

sns.scatterplot(data=final, x=range(final.shape[0]), y="scores", hue="from_prox") \
   .set_title(f"Scores")

# for t in np.linspace(1, 10, 10):
#     print(t)
#     outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
#     print(f1_score(y_train.values, outlier_scores_thresholded))
#     print("___")
plt.show()


















