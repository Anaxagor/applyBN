import json

from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore

import numpy as np

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import f1_score
import pandas as pd

# np.random.seed(20)
# Ecoli dataset
from ucimlrepo import fetch_ucirepo
ecoli = fetch_ucirepo(id=39)
df = ecoli.data.features
y = ecoli.data.targets
# Among the 8 classes omL, imL, and imS are the minority classes and used as outliers
y = pd.DataFrame(np.where(np.isin(y, ["omL", "imL", "imS"]), 1, 0))
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

cv = 20
skf = StratifiedKFold(n_splits=cv)

i = -1
cross_val_scores = {"scores": []}
for train_indexes, test_indexes in skf.split(df, y):
    i += 1
    print(i)
    X_train, X_test = df.iloc[train_indexes, :], df.iloc[test_indexes, :]
    y_train, y_test = y.iloc[train_indexes, :], y.iloc[test_indexes, :]

    estimator = BNEstimator(has_logit=True,
                            use_mixture=True,
                            bn_type="cont")

    # encoder = pp.LabelEncoder()
    discretizer = pp.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')

    # create a preprocessor object with encoder and discretizer
    p = Preprocessor([('discretizer', discretizer)])
    # discretize data for structure learning
    discretized_data, encoding = p.apply(X_train)

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
    PROX_STEPS = 45
    # score = ModelBasedScore(estimator)
    score_proximity = LocalOutlierScore()
    score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=PROX_STEPS)

    detector = TabularDetector(estimator,
                               score=score,
                               target_name=None)

    detector.fit(discretized_data, y=None,
                 clean_data=X_train, descriptor=info,
                 inject=False, bn_params={"scoring_function": ("K2",),
                                          "progress_bar": False})

    # detector.estimator.bn.get_info(as_df=False)

    outlier_scores = detector.detect(X_test, return_scores=True)

    final = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1), y_test.values.reshape(-1, 1).astype(int)]),
                         columns=["score", "anomaly"])

    thresholds = np.linspace(1, outlier_scores.max(), 100)
    eval_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
        eval_scores.append(f1_score(y_test.values, outlier_scores_thresholded))

    # plt.figure(figsize=(20, 12))
    # desc = f"""[Nonlinear, no scaler, model metric:Z-score, prox_step: {PROX_STEPS}]"""
    # sns.scatterplot(data=final, x=range(final.shape[0]), s=20,
    #                 y="score", hue="anomaly") \
    #     .set_title("Scores; Impacts(P, M): "
    #                   f"[{detector.score.proximity_impact.round(3)}, {detector.score.model_impact.round(3)}]")

    # plt.figure()
    # ax = sns.lineplot(x=thresholds, y=eval_scores)
    # ax.set(xlabel='thresholds', ylabel='f1_score', title=f"{i}: sensitivity analysis")
    # plt.show()
    cross_val_scores["scores"].append(np.max(eval_scores))
# from sklearn.cluster import DBSCAN, KMeans

# cls = KMeans(n_clusters=2, random_state=0,
#              init=np.array([[300, 60], [300, 10]]))
# cls = DBSCAN(eps=1)
#
# labels = cls.fit_predict(
#     X=np.asarray(
#         [[i, j] for i, j in enumerate(final["score"].values)])
# )
#
# indexes = np.where(labels == 1)[0]
#
# print(np.unique(labels))
# if np.unique(labels).size == 1:
#     raise Exception("1 index")
# elif np.unique(labels).size == 2:
#     sns.scatterplot(x=indexes, y=final["score"][indexes], color="r")
# else:
#     sns.scatterplot(x=indexes, y=final["score"][indexes], hue=labels)

# plt.show()

# plt.savefig(f"real_results/cardio/{desc}.png")

with open(f"ecoli_cv/2cross_val_with_skf_cv{cv}_mixture.json", "w+") as f:
    json.dump(cross_val_scores, f)
