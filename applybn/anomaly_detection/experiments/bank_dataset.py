from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import seaborn as sns
from sklearn.metrics import f1_score

df = pd.read_csv("data/tabular/bank_data.csv", index_col=0) \
    .sample(3000, random_state=42, ignore_index=True)

disc_cols = df.select_dtypes(include=["object"]).columns
cont_cols = df.select_dtypes(include=["int64"]).columns
df[cont_cols] = df[cont_cols].astype(float)
y = pd.DataFrame(df.pop('y'))

y_coder = pp.LabelEncoder()
y = pd.DataFrame(y_coder.fit_transform(y))
print(dict(zip(y_coder.classes_, range(len(y_coder.classes_)))))
print(df.shape)

skf = StratifiedShuffleSplit(n_splits=10)
i = -1
cross_val_scores = {"scores": []}
for train_indexes, test_indexes in skf.split(df, y):
    i += 1
    print(i)
    X_train, X_test = df.iloc[train_indexes, :], df.iloc[test_indexes, :]
    y_train, y_test = y.iloc[train_indexes, :], y.iloc[test_indexes, :]

    estimator = BNEstimator(has_logit=True,
                            use_mixture=False,
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
    # detector.estimator.bn.get_info(as_df=False)

    outlier_scores = detector.detect(X_test, return_scores=True)

    # final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1),
    #                                 y_test.values.reshape(-1, 1).astype(int)]),
    #                      columns=["score", "anomaly"])

    thresholds = np.linspace(1, outlier_scores.max(), 100)
    eval_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
        eval_scores.append(f1_score(y_test.values, outlier_scores_thresholded))

    # desc = f"""[Nonlinear, no scaler, model metric:Z-score, summation]"""
    # sns.scatterplot(data=final, x=range(final.shape[0]),
    #                 y="score", hue="anomaly") \
    #     .set_title("Scores; Impacts(P, M): "
    #                   f"[{detector.score.proximity_impact.round(3)}, {detector.score.model_impact.round(3)}]")

    plt.figure()
    ax = sns.lineplot(x=thresholds, y=eval_scores)
    ax.set(xlabel='thresholds', ylabel='f1_score', title=f"{i}: sensitivity analysis")
    plt.show()
    # plt.savefig(f"real_results/{desc}.png")