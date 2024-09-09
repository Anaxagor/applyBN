from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
from applybn.anomaly_detection.scores.proximity_based import LocalOutlierScore
from applybn.anomaly_detection.scores.mixed import ODBPScore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
import seaborn as sns
import json
from sklearn.linear_model import LinearRegression

# np.random.seed(20)

MEAN_F1, VAR_F1 = 4.5, 10
MEAN_F2, VAR_F2 = 3.5, 5
A, B = 5, 10

PROX_SHIFTS = {"loc": 2000, "scale": 100}
ANOMALY_PROBA = 0.05


def f(x1, x2, a, b):
    # return a * x1 + b * x2 + a * b
    return a * x1 ** 2 + b * x2 + a * b


def bomb_prox(df, shifts_params,
              anomaly_proba,
              target="feature3", out_name="anomaly"):
    shifts = np.random.normal(**shifts_params, size=1000)

    indexes_left = np.random.choice([1, 0], 1000, p=[anomaly_proba, 1 - anomaly_proba], )
    shifts[indexes_left == 0] = 0

    if all(indexes_left == 0):
        raise Exception("Bomb doesn't work.")

    df[target] += shifts
    df[out_name] = indexes_left
    return df


def bomb_model(df, target1="feature2", target2="feature3", out_name="anomaly"):
    number_of_anomaly_points = 30

    df_ = df.copy()

    aux_model = LinearRegression()

    aux_model.fit(df[target1].to_numpy().reshape(-1, 1),
                  df[target2].to_numpy().reshape(-1, 1))

    anomaly_locs = df.sample(number_of_anomaly_points).index

    df_[target1][anomaly_locs] += 25

    anomaly_preds = aux_model.predict(df_[target1][anomaly_locs].to_numpy().reshape(-1, 1))

    noise = np.random.normal(loc=10, scale=500, size=(anomaly_preds.shape[0], 1))
    anomaly_preds += noise

    df_[target2][anomaly_locs] = anomaly_preds.flatten()

    anomaly_label = np.zeros(shape=(1000, 1))
    anomaly_label[anomaly_locs] = 1

    df_[out_name] = anomaly_label
    return df_


N = 1
final = {"scores": {"f1": []}}
for n in range(N):
    print(n)
    df = pd.DataFrame({"feature1": np.random.normal(scale=VAR_F1, size=1000, loc=MEAN_F1),
                       "feature2": np.random.normal(scale=VAR_F2, size=1000, loc=MEAN_F2),
                       "feature4": np.random.normal(scale=A, size=1000, loc=B),
                       })

    df["feature3"] = f(df["feature1"], df["feature2"], A, B)
    df = bomb_prox(df, shifts_params=PROX_SHIFTS, anomaly_proba=ANOMALY_PROBA)

    # df = bomb_model(df, out_name="anomaly1")
    # df = bomb_prox(df, out_name="anomaly2")
    # df["anomaly"] = np.max(df[["anomaly1", "anomaly2"]], axis=1)
    # df.drop(["anomaly1", "anomaly2"], axis=1, inplace=True)

    plt.figure()
    g = sns.PairGrid(data=df, hue="anomaly", corner=True)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.scatterplot)

    plt.savefig("tmp.png")

    y_train = df.pop("anomaly")

    estimator = BNEstimator(has_logit=True,
                            bn_type="cont")

    # encoder = pp.LabelEncoder()
    discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

    # create a preprocessor object with encoder and discretizer
    p = Preprocessor([('discretizer', discretizer)])
    # p = Preprocessor([])
    discretized_data, encoding = p.apply(df)
    info = p.info

    score_proximity = LocalOutlierScore(n_neighbors=20)
    score = ODBPScore(estimator, score_proximity, encoding=encoding, proximity_steps=2)

    detector = TabularDetector(estimator,
                               score=score,
                               target_name=None)

    detector.estimator.bn.add_nodes(info)
    detector.estimator.bn.set_structure(edges=[["feature1", 'feature3'], ["feature2", "feature3"]])
    detector.estimator.bn.fit_parameters(df)

    # detector.fit(discretized_data, y=None,
    #              clean_data=df, descriptor=info,
    #              inject=False,
    #              bn_params={"scoring_function": ("BIC",),
    #                         "progress_bar": False},
    #              )

    # detector.estimator.bn.get_info(as_df=False)
    outlier_scores = detector.detect(df, return_scores=True)

    final_ = pd.DataFrame(np.hstack([outlier_scores.values.reshape(-1, 1),
                                    y_train.values.reshape(-1, 1)]),
                         columns=["scores", "anomaly"])

    # print(
    #     f"Model impact: {detector.score.model_impact} \n"
    #     f"Proximity impact: {detector.score.proximity_impact}")

    # desc = f"""[Linear, no scaler, model metric:Z-score, summation]"""

    plt.figure()
    sns.scatterplot(data=final_, x=range(final_.shape[0]),
                    y="scores", hue="anomaly") \
        .set_title("Scores; Impacts(P, M): "
                   f"[{detector.score.proximity_impact.round(3)}, {detector.score.model_impact.round(3)}]")

    # plt.savefig(f"synth_results/{desc}.png")
    plt.show()

    thresholds = np.linspace(1, outlier_scores.max(), 100)
    f1_scores = []

    for t in thresholds:
        outlier_scores_thresholded = np.where(outlier_scores < t, 0, 1)
        f1_scores.append(f1_score(y_train.values, outlier_scores_thresholded))

    final["scores"]["f1"].append(np.max(f1_scores))

print(final['scores']["f1"])
print(f"Mean Score: {np.mean(final['scores']['f1']).round(3)}\n"
      f"Std: {np.std(final['scores']['f1']).round(3)}")

# print(f"Mean recall: {np.mean(final['scores']['recall']).round(3)}\n")
# print(f"Mean precision: {np.mean(final['scores']['precision']).round(3)}\n")

# with open("exp1.3(nonlinear, K2, more metrics).json", "w+") as f:
#     json.dump(final, f)

# plt.figure()
# ax = sns.lineplot(x=thresholds, y=f1_scores)
# ax.set(xlabel='thresholds', ylabel='f1_score', title="sensitivity analysis")
# plt.grid()
# plt.show()
