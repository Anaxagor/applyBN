from applybn.anomaly_detection.static_anomaly_detector.tabular_detector import TabularDetector
from applybn.core.estimators import BNEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from bamt.preprocessors import Preprocessor
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split


# X = pd.read_csv("../../data/benchmarks/bank_data/X_data.csv", index_col=0)
# y = pd.read_csv("../../data/benchmarks/bank_data/target.csv", index_col=0)
# print(X.shape)
# print(y.shape)
df = pd.read_csv("../../data/benchmarks/bank_data/bank_data.csv", index_col=0).sample(3000, random_state=42)
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["y"], axis=1), df["y"], test_size=0.5, random_state=42, stratify=df["y"])
# SAMPLE_SIZE = 5000
#
# X = pd.concat([X, y], axis=1).dropna().sample(SAMPLE_SIZE).astype("object")
# y = X.pop("y")

estimator = BNEstimator(has_logit=True,
                        bn_type="disc")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('encoder', encoder)])
# discretize data for structure learning

discretized_data, est = p.apply(X_train)
disc_y = pp.LabelEncoder().fit_transform(y_train)

# get information about data
info = p.info

detector = TabularDetector(estimator,
                           target_name="y")

detector.fit(discretized_data, y=disc_y,
             clean_data=X_train, descriptor=info,
             how="inject")
detector.bn.get_info(as_df=False)

preds = detector.predict_anomaly(X_test.astype(str))
plt.hist(preds)
plt.show()

preds_trunc = np.where(preds > 0.7, 0, 1)
print(
    np.unique(preds_trunc, return_counts=True)
)
# print(
#     detector.bn.find_family("y", height=3, depth=3, plot_to="tt5.html")
# )
