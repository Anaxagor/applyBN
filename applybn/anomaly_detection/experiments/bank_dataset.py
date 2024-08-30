from sklearn.preprocessing import StandardScaler

from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("../../../data/benchmarks/bank_data/bank_data.csv", index_col=0) \
    .sample(3000, random_state=42, ignore_index=True)

disc_cols = df.select_dtypes(include=["object"]).columns
cont_cols = df.select_dtypes(include=["int64"]).columns
df[cont_cols] = df[cont_cols].astype(float)

for col in cont_cols:
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

print(df.describe().loc[['min', 'max']])

# print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["y"], axis=1), df["y"], test_size=0.5, random_state=42, stratify=df["y"])

estimator = BNEstimator(has_logit=False,
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
score_proximity = LocalOutlierScore(n_neighbors=30)
score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=5)

detector = TabularDetector(estimator,
                           score=score,
                           target_name=None)

detector.fit(discretized_data, y=None,
             clean_data=X_train, descriptor=info,
             inject=False)
detector.estimator.bn.get_info(as_df=False)


outlier_scores = detector.detect(X_train, return_scores=True)

final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1), disc_y.values.reshape(-1, 1).astype(int)]),
                     columns=["score", "anomaly"])


desc = f"""[Nonlinear, no scaler, model metric:Z-score, summation]"""
sns.scatterplot(data=final, x=range(final.shape[0]),
                y="score", hue="anomaly") \
    .set_title("Scores; Impacts(P, M): "
                  f"[{detector.score.proximity_impact.round(3)}, {detector.score.model_impact.round(3)}]")
plt.show()
# plt.savefig(f"real_results/{desc}.png")