# Example for RegressorExplainer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from applybn.explainable.causal_explain import RegressorExplainer, ClassifierExplainer

# Generate a synthetic dataset for regression
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
data['treatment'] = np.random.rand(100)  # Add a continuous treatment variable
data['outcome'] = y

treatment = 'treatment'
outcome = 'outcome'
common_causes = [f'feature_{i}' for i in range(5)]

# Initialize and test the RegressorExplainer
base_algo = LinearRegression()
regressor_explainer = RegressorExplainer(base_algo, data, treatment, outcome, common_causes)
predictions = regressor_explainer.fit_predict()
print("Feature Importance (Regression):", regressor_explainer.get_feature_importance())
print("SHAP Values (Regression):", regressor_explainer.get_shap_values())
print("Confidences (Regression):", regressor_explainer.get_confidences())
print("Uncertainties (Regression):", regressor_explainer.get_uncertainties())
regressor_explainer.plot_feature_importance()
regressor_explainer.plot_shap_values()
regressor_explainer.plot_confidences()
regressor_explainer.plot_uncertainties()

# Build causal graph with confidence
causal_effects_reg = regressor_explainer.build_causal_graph_with_confidence()
print("Causal Effects on Confidence (Regression):", causal_effects_reg)

# Example for ClassifierExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a synthetic dataset for classification
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
data['treatment'] = np.random.randint(2, size=100)  # Add a binary treatment variable
data['outcome'] = y

treatment = 'treatment'
outcome = 'outcome'
common_causes = [f'feature_{i}' for i in range(5)]

# Initialize and test the ClassifierExplainer
base_algo = LogisticRegression()
classifier_explainer = ClassifierExplainer(base_algo, data, treatment, outcome, common_causes)
predictions_classes = classifier_explainer.fit_predict()
print("Feature Importance (Classification):", classifier_explainer.get_feature_importance())
print("SHAP Values (Classification):", classifier_explainer.get_shap_values())
print("Confidences (Classification):", classifier_explainer.get_confidences())
print("Uncertainties (Classification):", classifier_explainer.get_uncertainties())
classifier_explainer.plot_feature_importance()
classifier_explainer.plot_shap_values()
classifier_explainer.plot_confidences()
classifier_explainer.plot_uncertainties()

# Build causal graph with confidence
causal_effects_cls = classifier_explainer.build_causal_graph_with_confidence()
print("Causal Effects on Confidence (Classification):", causal_effects_cls)
