import numpy as np
from sklearn.model_selection import KFold

from applybn.explainable.causal_explainer import Explainer


class RegressorExplainer(Explainer):
    """
    Explainer for regression models.

    Example:
        >>> base_algo = LinearRegression()
        >>> data = pd.DataFrame(np.random.randn(100, 7), columns=[f'feature_{i}' for i in range(5)] + ['treatment', 'outcome'])
        >>> treatment = 'treatment'
        >>> outcome = 'outcome'
        >>> common_causes = [f'feature_{i}' for i in range(5)]
        >>> explainer = RegressorExplainer(base_algo, data, treatment, outcome, common_causes)
        >>> predictions = explainer.fit_predict()
        >>> print(explainer.get_feature_importance())
        >>> explainer.plot_feature_importance()
    """

    def _calculate_confidence_uncertainty(self):
        """
        Calculates confidence and uncertainty using cross-validation for regression models.
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

            # Calculate residuals
            residuals = test_data[self.outcome] - preds

            # Normalize residuals
            max_abs_residual = np.max(np.abs(residuals))
            normalized_residuals = np.abs(residuals) / max_abs_residual

            # Calculate confidence
            conf = 1 - normalized_residuals
            confidence_list.extend(conf)

            # Calculate uncertainty (standard deviation of residuals)
            uncertainty_list.extend(np.std(residuals) * np.ones(len(test_data)))

        self.predictions = predictions
        self.confidences = np.array(confidence_list)
        self.uncertainties = np.array(uncertainty_list)
