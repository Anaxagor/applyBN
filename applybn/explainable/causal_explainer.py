import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
import shap
import dowhy
from dowhy import CausalModel
import statsmodels.api as sm


class Explainer(ABC):
    def __init__(
        self,
        base_algo: BaseEstimator,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        common_causes: list,
        n_splits: int = 5,
    ):
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
            data=data, treatment=treatment, outcome=outcome, common_causes=common_causes
        )

    def fit_predict(self):
        # Identify causal effect
        identified_estimand = self.causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )
        # Estimate causal effect
        causal_estimate = self.causal_model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression"
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
        pass

    def get_feature_importance(self):
        if hasattr(self.model, "params"):
            self.feature_importance_ = np.abs(
                self.model.params[1:]
            )  # Exclude the intercept
        else:
            raise NotImplementedError(
                "Feature importance is not implemented for this model type."
            )
        return self.feature_importance_

    def get_shap_values(self):
        def model_predict(data):
            data = sm.add_constant(data)
            return self.model.predict(data)

        explainer = shap.Explainer(
            model_predict, self.data[self.common_causes + [self.treatment]]
        )
        self.shap_values_ = explainer(self.data[self.common_causes + [self.treatment]])
        return self.shap_values_

    def plot_feature_importance(self):
        if self.feature_importance_ is None:
            self.get_feature_importance()
        plt.bar(self.common_causes + [self.treatment], self.feature_importance_)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.show()

    def plot_shap_values(self):
        if self.shap_values_ is None:
            self.get_shap_values()
        shap.summary_plot(
            self.shap_values_, self.data[self.common_causes + [self.treatment]]
        )

    def plot_confidences(self):
        if self.confidences is None:
            raise ValueError("Confidences have not been calculated.")
        plt.scatter(self.predictions, self.confidences)
        plt.xlabel("Predictions")
        plt.ylabel("Confidences")
        plt.title("Confidences vs Predictions")
        plt.show()

    def plot_uncertainties(self):
        if self.uncertainties is None:
            raise ValueError("Uncertainties have not been calculated.")
        plt.scatter(self.predictions, self.uncertainties)
        plt.xlabel("Predictions")
        plt.ylabel("Uncertainties")
        plt.title("Uncertainties vs Predictions")
        plt.show()

    def get_confidences(self):
        return self.confidences

    def get_uncertainties(self):
        return self.uncertainties


class RegressorExplainer(Explainer):
    def _calculate_confidence_uncertainty(self):
        # Calculate confidence and uncertainty using cross-validation
        kf = KFold(n_splits=self.n_splits)
        predictions = np.zeros(len(self.data))
        uncertainty_list = []
        confidence_list = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = self.base_algo.fit(
                train_data[self.common_causes + [self.treatment]],
                train_data[self.outcome],
            )
            preds = model.predict(test_data[self.common_causes + [self.treatment]])
            predictions[test_index] = preds

            residuals = test_data[self.outcome] - preds
            conf = 1 - (np.abs(residuals) / np.max(np.abs(residuals)))
            confidence_list.extend(conf)
            uncertainty_list.extend(np.std(residuals) * np.ones(len(test_data)))

        self.predictions = predictions
        self.confidences = np.array(confidence_list)
        self.uncertainties = np.array(uncertainty_list)


class ClassifierExplainer(Explainer):
    def _calculate_confidence_uncertainty(self):
        # Calculate confidence and uncertainty using cross-validation
        kf = KFold(n_splits=self.n_splits)
        predictions = np.zeros(len(self.data))
        uncertainty_list = []
        confidence_list = []

        for train_index, test_index in kf.split(self.data):
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]

            model = self.base_algo.fit(
                train_data[self.common_causes + [self.treatment]],
                train_data[self.outcome],
            )
            preds = model.predict(test_data[self.common_causes + [self.treatment]])
            predictions[test_index] = preds

            if hasattr(model, "predict_proba"):
                proba_preds = model.predict_proba(
                    test_data[self.common_causes + [self.treatment]]
                )
                conf = np.max(proba_preds, axis=1)
            else:
                residuals = test_data[self.outcome] - preds
                conf = 1 - (np.abs(residuals) / np.max(np.abs(residuals)))
            confidence_list.extend(conf)
            residuals = test_data[self.outcome] - preds
            uncertainty_list.extend(np.std(residuals) * np.ones(len(test_data)))

        self.predictions = predictions
        self.confidences = np.array(confidence_list)
        self.uncertainties = np.array(uncertainty_list)


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a synthetic dataset for classification
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
data["treatment"] = np.random.randint(2, size=100)  # Add a binary treatment variable
data["outcome"] = y

treatment = "treatment"
outcome = "outcome"
common_causes = [f"feature_{i}" for i in range(5)]

# Initialize and test the ClassifierExplainer
base_algo = LogisticRegression()
classifier_explainer = ClassifierExplainer(
    base_algo, data, treatment, outcome, common_causes
)
predictions = classifier_explainer.fit_predict()
print(
    "Feature Importance (Classification):",
    classifier_explainer.get_feature_importance(),
)
print("SHAP Values (Classification):", classifier_explainer.get_shap_values())
print("Confidences (Classification):", classifier_explainer.get_confidences())
print("Uncertainties (Classification):", classifier_explainer.get_uncertainties())
classifier_explainer.plot_feature_importance()
classifier_explainer.plot_shap_values()
classifier_explainer.plot_confidences()
classifier_explainer.plot_uncertainties()
