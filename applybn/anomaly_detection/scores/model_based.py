from applybn.anomaly_detection.scores.score import Score
import pandas as pd
import numpy as np


class ModelBasedScore(Score):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def score(self, X):
        probas = self.model.predict_proba(X)

        if isinstance(probas, pd.Series):
            return probas.values
        if isinstance(probas, np.ndarray):
            return probas
