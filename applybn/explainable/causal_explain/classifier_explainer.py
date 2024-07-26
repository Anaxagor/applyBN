import numpy as np
from sklearn.model_selection import KFold

from applybn.explainable.causal_explainer import Explainer


class ClassifierExplainer(Explainer):
    """
    Explainer for classification models.

    Example:
        >>> base_algo = LogisticRegression()
        >>> data = pd.DataFrame(np.random.randn(100, 7), columns=[f'feature_{i}' for i in range(5)] + ['treatment', 'outcome'])
        >>> treatment = 'treatment'
        >>> outcome = 'outcome'
        >>> common_causes = [f'feature_{i}' for i in range(5)]
        >>> explainer = ClassifierExplainer(base_algo, data, treatment, outcome, common_causes)
        >>> predictions = explainer.fit_predict()
        >>> print(explainer.get_feature_importance())
        >>> explainer.plot_feature_importance()
    """

    def _calculate_confidence_uncertainty(self):
        """
        Calculates confidence and uncertainty using cross-validation for classification models.
        """
        kf = KFold(n_splits=self.n_splits)
        predictions = np.zeros(len(self.data))
        uncertainty_list = []
        confidence_list = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            # Fit the model
            model = self.base_algo.fit(train_data[self.common_causes + [self.treatment]], train_data[self.outcome])
            preds = model.predict(test_data[self.common_causes + [self.treatment]])
            predictions[test_index] = preds

            # Calculate confidence using predict_proba if available
            if hasattr(model, 'predict_proba'):
                proba_preds = model.predict_proba(test_data[self.common_causes + [self.treatment]])
                conf = np.max(proba_preds, axis=1)
            else:
                residuals = test_data[self.outcome] - preds
                conf = 1 - (np.abs(residuals) / np.max(np.abs(residuals)))
            confidence_list.extend(conf)

            # Calculate uncertainty (standard deviation of residuals)
            residuals = test_data[self.outcome] - preds
            uncertainty_list.extend(np.std(residuals) * np.ones(len(test_data)))

        self.predictions = predictions
        self.confidences = np.array(confidence_list)
        self.uncertainties = np.array(uncertainty_list)
