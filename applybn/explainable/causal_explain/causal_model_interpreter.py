# Import necessary libraries
from data_iq import DataIQ_SKLearn
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

class ModelInterpreterEconML:
    """
    A class to interpret machine learning model results using causal inference with EconML.
    """

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.dataiq = None
        self.confidence = None
        self.aleatoric_uncertainty = None
        self.feature_impact = None

    def compute_confidence_uncertainty(self):
        self.dataiq = DataIQ_SKLearn(X=self.X_train, y=self.y_train)
        self.model.fit(self.X_train, self.y_train)
        nest = getattr(self.model, 'n_estimators', 1)
        for i in range(1, nest + 1):
            self.dataiq.on_epoch_end(clf=self.model, iteration=i)
        self.confidence = self.dataiq.confidence
        self.aleatoric_uncertainty = self.dataiq.aleatoric

    def estimate_feature_impact(self):
        if self.confidence is None:
            raise ValueError("Confidence metrics not computed. Call compute_confidence_uncertainty() first.")

        self.feature_impact = {}

        if not isinstance(self.X_train, pd.DataFrame):
            X = pd.DataFrame(self.X_train)
        else:
            X = self.X_train.copy()

        y = self.confidence

        for feature in X.columns:
            print(f"Estimating impact for feature: {feature}")
            T = X[[feature]].values.ravel()
            covariates = X.drop(columns=[feature]).values

            # Instantiate the CausalForestDML estimator
            model_y = RandomForestRegressor(n_estimators=100, random_state=42)
            model_t = RandomForestRegressor(n_estimators=100, random_state=42)
            dml = CausalForestDML(model_y=model_y, model_t=model_t, random_state=42)

            # Fit the DML model
            dml.fit(Y=y, T=T, X=covariates)

            # Estimate the treatment effect
            te = dml.effect(X=covariates)

            # Store the average treatment effect for the feature
            self.feature_impact[feature] = np.mean(te)

    def get_feature_impact(self):
        if self.feature_impact is None:
            raise ValueError("Feature impacts not computed. Call estimate_feature_impact() first.")
        return self.feature_impact

    def display_feature_impact(self):
        if self.feature_impact is None:
            raise ValueError("Feature impacts not computed. Call estimate_feature_impact() first.")
        print("\nFeature Impact on Model Confidence (EconML):")
        for feature, impact in self.feature_impact.items():
            print(f"- {feature}: {impact:.4f}")

if __name__ == "__main__":
    # Import necessary libraries
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the machine learning model
    clf = xgb.XGBClassifier(n_estimators=10, random_state=42)

    # Create an instance of ModelInterpreterEconML
    interpreter_econml = ModelInterpreterEconML(model=clf, X_train=X_train, y_train=y_train)

    # Compute confidence and uncertainty
    interpreter_econml.compute_confidence_uncertainty()

    # Estimate feature impact on confidence
    interpreter_econml.estimate_feature_impact()

    # Retrieve and display feature impacts
    feature_impacts_econml = interpreter_econml.get_feature_impact()
    interpreter_econml.display_feature_impact()
