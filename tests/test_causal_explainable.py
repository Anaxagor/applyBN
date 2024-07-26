import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification

from applybn.explainable.causal_explain import ClassifierExplainer
from applybn.explainable.causal_explain import RegressorExplainer


class TestExplainers(unittest.TestCase):

    def setUp(self):
        # Setup common data for regression and classification tests
        self.regression_data = pd.DataFrame(
            np.random.randn(100, 7),
            columns=[f"feature_{i}" for i in range(5)] + ["treatment", "outcome"],
        )
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
        self.regression_data.iloc[:, :5] = X
        self.regression_data["treatment"] = np.random.rand(100)
        self.regression_data["outcome"] = y

        self.classification_data = pd.DataFrame(
            np.random.randn(100, 7),
            columns=[f"feature_{i}" for i in range(5)] + ["treatment", "outcome"],
        )
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.classification_data.iloc[:, :5] = X
        self.classification_data["treatment"] = np.random.randint(2, size=100)
        self.classification_data["outcome"] = y

    def test_regressor_explainer(self):
        treatment = "treatment"
        outcome = "outcome"
        common_causes = [f"feature_{i}" for i in range(5)]

        # Initialize and test the RegressorExplainer
        base_algo = LinearRegression()
        regressor_explainer = RegressorExplainer(
            base_algo, self.regression_data, treatment, outcome, common_causes
        )
        predictions = regressor_explainer.fit_predict()

        self.assertEqual(len(predictions), 100)
        self.assertIsNotNone(regressor_explainer.get_feature_importance())
        self.assertIsNotNone(regressor_explainer.get_shap_values())
        self.assertIsNotNone(regressor_explainer.get_confidences())
        self.assertIsNotNone(regressor_explainer.get_uncertainties())

        # Plotting (not asserting here, just ensuring no exceptions)
        regressor_explainer.plot_feature_importance()
        regressor_explainer.plot_shap_values()
        regressor_explainer.plot_confidences()
        regressor_explainer.plot_uncertainties()

    def test_classifier_explainer(self):
        treatment = "treatment"
        outcome = "outcome"
        common_causes = [f"feature_{i}" for i in range(5)]

        # Initialize and test the ClassifierExplainer
        base_algo = LogisticRegression()
        classifier_explainer = ClassifierExplainer(
            base_algo, self.classification_data, treatment, outcome, common_causes
        )
        predictions = classifier_explainer.fit_predict()

        self.assertEqual(len(predictions), 100)
        self.assertIsNotNone(classifier_explainer.get_feature_importance())
        self.assertIsNotNone(classifier_explainer.get_shap_values())
        self.assertIsNotNone(classifier_explainer.get_confidences())
        self.assertIsNotNone(classifier_explainer.get_uncertainties())

        # Plotting (not asserting here, just ensuring no exceptions)
        classifier_explainer.plot_feature_importance()
        classifier_explainer.plot_shap_values()
        classifier_explainer.plot_confidences()
        classifier_explainer.plot_uncertainties()


if __name__ == "__main__":
    unittest.main()
