import pandas as pd
import numpy as np

from applybn.core.estimators import BNEstimator
from bamt.preprocessors import Preprocessor

from sklearn import preprocessing as pp
# from sklearn.pipeline import Pipeline

# X = bank_marketing.data.features.drop(["poutcome", "contact"], axis=1).dropna()
# y = bank_marketing.data.targets.dropna()

np.random.seed(50)
X = pd.read_csv("../../data/benchmarks/bank_data/X_data.csv", index_col=0)
y = pd.read_csv("../../data/benchmarks/bank_data/target.csv", index_col=0)

SAMPLE_SIZE = 1000

X = pd.concat([X, y], axis=1).dropna().sample(SAMPLE_SIZE).astype("object")
# y = X.pop("y")

# df = pd.concat([X, y], axis=1).dropna().sample(100)
# print(df.dtypes)
# print(df.shape)


estimator = BNEstimator(has_logit=True,
                        has_continuous_data=False)


encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# create a preprocessor object with encoder and discretizer
p = Preprocessor([('encoder', encoder)])
# inverse_p = [("inverse_encoder", )]
# discretize data for structure learning
discretized_data, est = p.apply(X)

# get information about data
info = p.info

estimator.fit(discretized_data,
              clean_data=X, descriptor=info, partial=True,
              params={"bl_add": [("y", node_name) for node_name in discretized_data.columns][:-1]})

estimator.bn.get_info(as_df=False)

print(
    estimator.bn.find_family("y", height=3, depth=3, plot_to="with_y.html")
)
# detector.bn.plot("tt5.html")
# print(detector.predict_proba(discretized_data.drop("y", axis=1)))