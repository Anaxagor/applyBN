import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import shap
from dowhy import CausalModel
import statsmodels.api as sm


class Explainer(ABC):
    """
    Abstract base class for explaining causal models.

    Attributes:
        base_algo (BaseEstimator): The base machine learning algorithm.
        data (pd.DataFrame): The dataset containing features, treatment, and outcome.
        treatment (str): The name of the treatment variable.
        outcome (str): The name of the outcome variable.
        common_causes (list): List of common cause variables.
        n_splits (int): Number of splits for K-Fold cross-validation.
        model (any): The trained causal model.
        feature_importance_ (np.ndarray): Array to store feature importance values.
        shap_values_ (any): Object to store SHAP values.
        predictions (np.ndarray): Array to store model predictions.
        confidences (np.ndarray): Array to store confidence values.
        uncertainties (np.ndarray): Array to store uncertainty values.

    Methods:
        fit_predict(): Identifies, estimates, and predicts using the causal model.
        _calculate_confidence_uncertainty(): Abstract method to calculate confidence and uncertainty.
        get_feature_importance(): Calculates and returns feature importance.
        get_shap_values(): Calculates and returns SHAP values.
        plot_feature_importance(): Plots the feature importance.
        plot_shap_values(): Plots the SHAP values.
        plot_confidences(): Plots the confidences vs. predictions.
        plot_uncertainties(): Plots the uncertainties vs. predictions.
        get_confidences(): Returns the confidence values.
        get_uncertainties(): Returns the uncertainty values.

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

    def __init__(self, base_algo: BaseEstimator, data: pd.DataFrame, treatment: str, outcome: str, common_causes: list,
                 n_splits: int = 5):
        self.base_algo = base_algo
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self.n_splits = n_splits
        self.model = None
        self.feature_importance_ = None
        self.shap_values_ = None
        self.predictions = None
        self.confidences = None
        self.uncertainties = None

        # Initialize the causal model
        self.causal_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes
        )

    def fit_predict(self):
        """
        Identifies, estimates, and predicts using the causal model.

        Returns:
            np.ndarray: The predicted values.
        """
        # Identify causal effect
        identified_estimand = self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        # Estimate causal effect
        causal_estimate = self.causal_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )
        self.model = causal_estimate.estimator.model
        # Make predictions using both common causes and the treatment variable
        predictors = self.data[self.common_causes + [self.treatment]]
        predictors = sm.add_constant(predictors)  # Add intercept term
        self.predictions = self.model.predict(predictors)
        self._calculate_confidence_uncertainty()
        return self.predictions

    @abstractmethod
    def _calculate_confidence_uncertainty(self):
        """
        Abstract method to calculate confidence and uncertainty.
        Must be implemented by subclasses.
        """
        pass

    def get_feature_importance(self):
        """
        Calculates and returns feature importance.

        Returns:
            np.ndarray: Feature importance values.
        """
        if hasattr(self.model, 'params'):
            self.feature_importance_ = np.abs(self.model.params[1:])  # Exclude the intercept
        else:
            raise NotImplementedError("Feature importance is not implemented for this model type.")
        return self.feature_importance_

    def get_shap_values(self):
        """
        Calculates and returns SHAP values.

        Returns:
            any: SHAP values object.
        """

        def model_predict(data):
            data = sm.add_constant(data)
            return self.model.predict(data)

        explainer = shap.Explainer(model_predict, self.data[self.common_causes + [self.treatment]])
        self.shap_values_ = explainer(self.data[self.common_causes + [self.treatment]])
        return self.shap_values_

    def plot_feature_importance(self):
        """
        Plots the feature importance.
        """
        if self.feature_importance_ is None:
            self.get_feature_importance()
        plt.bar(self.common_causes + [self.treatment], self.feature_importance_)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()

    def plot_shap_values(self):
        """
        Plots the SHAP values.
        """
        if self.shap_values_ is None:
            self.get_shap_values()
        shap.summary_plot(self.shap_values_, self.data[self.common_causes + [self.treatment]])

    def plot_confidences(self):
        """
        Plots the confidences vs. predictions.
        """
        if self.confidences is None:
            raise ValueError("Confidences have not been calculated.")
        plt.scatter(self.predictions, self.confidences)
        plt.xlabel('Predictions')
        plt.ylabel('Confidences')
        plt.title('Confidences vs Predictions')
        plt.show()

    def plot_uncertainties(self):
        """
        Plots the uncertainties vs. predictions.
        """
        if self.uncertainties is None:
            raise ValueError("Uncertainties have not been calculated.")
        plt.scatter(self.predictions, self.uncertainties)
        plt.xlabel('Predictions')
        plt.ylabel('Uncertainties')
        plt.title('Uncertainties vs Predictions')
        plt.show()

    def get_confidences(self):
        """
        Returns the confidence values.

        Returns:
            np.ndarray: Confidence values.
        """
        return self.confidences

    def get_uncertainties(self):
        """
        Returns the uncertainty values.

        Returns:
            np.ndarray: Uncertainty values.
        """
        return self.uncertainties
