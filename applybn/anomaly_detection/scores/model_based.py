from applybn.anomaly_detection.scores.score import Score


class ModelBasedScore(Score):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def score(self, X):
        return self.model.predict_proba(X)
