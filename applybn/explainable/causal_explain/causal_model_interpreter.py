from data_iq import DataIQ_SKLearn
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelInterpreterEconML:
    """
    A class to interpret machine learning model results using causal inference with EconML
    and measure how feature interventions impact model confidence.
    """

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None):
        """
        Initialize the model interpreter with data and model.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.dataiq = None
        self.confidence = None
        self.aleatoric_uncertainty = None
        self.feature_impact = None
        self.initial_feature_impact = None
        self.new_feature_impact = None

    def compute_confidence_uncertainty(self):
        """
        Compute model confidence and aleatoric uncertainty using the Data-IQ framework.
        """
        self.dataiq = DataIQ_SKLearn(X=self.X_train, y=self.y_train)
        self.model.fit(self.X_train, self.y_train)
        nest = getattr(self.model, 'n_estimators', 1)
        for i in range(1, nest + 1):
            self.dataiq.on_epoch_end(clf=self.model, iteration=i)
        self.confidence = self.dataiq.confidence
        self.aleatoric_uncertainty = self.dataiq.aleatoric

    def estimate_feature_impact(self):
        """
        Estimate the impact of each feature on model confidence using EconML's CausalForestDML.
        """
        if self.confidence is None:
            raise ValueError("Confidence metrics not computed. Call compute_confidence_uncertainty() first.")

        self.feature_impact = {}

        if not isinstance(self.X_train, pd.DataFrame):
            X = pd.DataFrame(self.X_train)
        else:
            X = self.X_train.copy()

        y = self.confidence

        for feature in X.columns:
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
        """
        Retrieve the feature impact values.
        """
        if self.feature_impact is None:
            raise ValueError("Feature impacts not computed. Call estimate_feature_impact() first.")
        return self.feature_impact

    def display_feature_impact(self):
        """
        Display the estimated feature impacts.
        """
        if self.feature_impact is None:
            raise ValueError("Feature impacts not computed. Call estimate_feature_impact() first.")
        print("\nFeature Impact on Model Confidence (EconML):")
        for feature, impact in self.feature_impact.items():
            print(f"- {feature}: {impact:.4f}")

    def intervene_and_plot_confidence_changes(self, top_k=5):
        """
        Perform interventions on the top K most impactful features and analyze how the confidence distribution changes.
        Additionally, plot the changes in feature impacts for intervened features.
        """
        # Sort features by impact
        if self.feature_impact is None:
            raise ValueError("Feature impacts not computed. Call estimate_feature_impact() first.")

        top_features = sorted(self.feature_impact, key=self.feature_impact.get, reverse=True)[:top_k]
        print(f"Top {top_k} features selected for interventions: {top_features}")

        initial_confidence = self.confidence.copy()

        # Store initial feature impact
        self.initial_feature_impact = {feature: self.feature_impact[feature] for feature in top_features}

        # Intervene on top features
        for feature in top_features:
            print(f"Intervening on feature: {feature}")
            # Perform simple intervention by adding a small value to the feature
            self.X_train[feature] += np.random.normal(0, 0.1, size=self.X_train.shape[0])

        # Recompute confidence after interventions
        self.compute_confidence_uncertainty()

        # Recompute feature impacts after intervention
        self.estimate_feature_impact()

        # Store new feature impact
        self.new_feature_impact = {feature: self.feature_impact[feature] for feature in top_features}

        new_confidence = self.confidence

        # Plot 1: Confidence distribution before and after intervention
        plt.figure(figsize=(10, 6))
        plt.hist(initial_confidence, bins=30, alpha=0.6, label='Before Interventions')
        plt.hist(new_confidence, bins=30, alpha=0.6, label='After Interventions')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution Before and After Interventions')
        plt.legend()
        plt.show()

        # Plot 2: Feature impact change only for intervened features
        feature_names = list(self.initial_feature_impact.keys())
        initial_impacts = list(self.initial_feature_impact.values())
        new_impacts = list(self.new_feature_impact.values())

        plt.figure(figsize=(10, 6))
        width = 0.4
        ind = np.arange(len(feature_names))

        plt.bar(ind - width / 2, initial_impacts, width, label='Initial Impact')
        plt.bar(ind + width / 2, new_impacts, width, label='New Impact')

        plt.xlabel('Intervened Features')
        plt.ylabel('Impact on Confidence')
        plt.title('Change in Feature Impacts on Confidence After Interventions')
        plt.xticks(ind, feature_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    import xgboost as xgb
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the machine learning model
    clf = xgb.XGBClassifier(n_estimators=10, random_state=42)

    # Create an instance of ModelInterpreterEconML
    interpreter = ModelInterpreterEconML(model=clf, X_train=X_train, y_train=y_train)

    # Step 1: Compute confidence and uncertainty
    interpreter.compute_confidence_uncertainty()

    # Step 2: Estimate feature impact on confidence
    interpreter.estimate_feature_impact()

    # Step 3: Retrieve and display feature impacts
    feature_impacts = interpreter.get_feature_impact()
    interpreter.display_feature_impact()

    # Step 4: Perform interventions on top features and plot confidence and impact changes
    interpreter.intervene_and_plot_confidence_changes(top_k=5)
