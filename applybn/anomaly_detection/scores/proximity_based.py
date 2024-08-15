from applybn.anomaly_detection.scores.score import Score
from sklearn.neighbors import LocalOutlierFactor


class LocalOutlierScore(Score):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs

    def score(self, X):
        clf = LocalOutlierFactor(**self.params)
        clf.fit(X)
        return clf.negative_outlier_factor_
