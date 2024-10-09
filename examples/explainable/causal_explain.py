import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

from applybn.explainable.causal_explain.data_iq import DataIQSKLearn
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, RandomForestClassifier
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class ModelInterpreter:
    def __init__(self, X_train, y_train, X_test, y_test, n_estimators=10):
        """
        Initialize the ModelInterpreter with training and test data.
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.n_estimators = n_estimators
        self.clf = None
        self.dataiq_train = None
        self.dataiq_test = None
        self.confidence_train = None
        self.confidence_test = None
        self.aleatoric_uncertainty_train = None
        self.aleatoric_uncertainty_test = None
        self.feature_effects = None
        self.confidence_test_before_intervention = None

    def train_model(self, model):
        """
        Train the model on the training data.
        """
        logging.info("Training the model.")
        self.clf = model
        self.clf.fit(self.X_train, self.y_train)

    def compute_confidence_uncertainty_train(self):
        """
        Compute model confidence and aleatoric uncertainty on training data using Data-IQ.
        """
        logging.info("Computing confidence and uncertainty on training data using Data-IQ.")
        self.dataiq_train = DataIQSKLearn(X=self.X_train, y=self.y_train)
        self.dataiq_train.on_epoch_end(clf=self.clf, iteration=self.n_estimators)
        self.confidence_train = self.dataiq_train.confidence
        self.aleatoric_uncertainty_train = self.dataiq_train.aleatoric

    def compute_confidence_uncertainty_test(self):
        """
        Compute model confidence and aleatoric uncertainty on test data using Data-IQ.
        """
        logging.info("Computing confidence and uncertainty on test data using Data-IQ.")
        self.dataiq_test = DataIQSKLearn(X=self.X_test, y=self.y_test)
        self.dataiq_test.on_epoch_end(clf=self.clf, iteration=self.n_estimators)
        self.confidence_test = self.dataiq_test.confidence
        self.aleatoric_uncertainty_test = self.dataiq_test.aleatoric

    def estimate_feature_impact(self):
        """
        Estimate the causal effect of each feature on the model's confidence using training data.
        """
        logging.info("Estimating feature impact using causal inference on training data.")
        self.feature_effects = {}
        for feature in self.X_train.columns:
            logging.info(f"Estimating effect of feature '{feature}'.")
            treatment = self.X_train[feature].values
            outcome = self.confidence_train
            covariates = self.X_train.drop(columns=[feature])

            # Use EconML's LinearDML estimator for continuous treatment
            est = CausalForestDML(model_y=RandomForestRegressor(),
                            model_t=RandomForestRegressor(),
                            discrete_treatment=False,
                            random_state=42)
            est.fit(Y=outcome, T=treatment, X=covariates)
            te = est.const_marginal_effect(covariates).mean()
            self.feature_effects[feature] = te

        # Convert to Series and sort
        self.feature_effects = pd.Series(self.feature_effects).sort_values(ascending=False)
        logging.info("Feature effects estimated.")

    def perform_intervention(self):
        """
        Perform an intervention on the top 5 most impactful features in the test data and observe changes.
        """
        if self.feature_effects is None:
            raise ValueError("Feature effects have not been estimated yet.")

        top_features = self.feature_effects.head(5).index.tolist()
        logging.info(f"Top {len(top_features)} most impactful features: {top_features}")

        # Compute confidence on test data before intervention
        self.compute_confidence_uncertainty_test()
        self.confidence_test_before_intervention = self.confidence_test.copy()

        # Store original feature values in test data
        original_feature_values_test = self.X_test[top_features].copy()

        # Plot original distributions and perform interventions on test data
        for feature in top_features:
            plt.figure(figsize=(10, 5))
            plt.hist(original_feature_values_test[feature], bins=30, alpha=0.5, label='Before Intervention')

            # Perform intervention: replace feature values with uniform random values within original range
            logging.info(f"Performing intervention on '{feature}' in test data.")
            min_val = original_feature_values_test[feature].min()
            max_val = original_feature_values_test[feature].max()
            np.random.seed(42)  # For reproducibility
            new_values = np.random.uniform(low=min_val, high=max_val, size=self.X_test.shape[0])
            self.X_test[feature] = new_values

            # Plot new distribution of the feature on the same plot
            plt.hist(self.X_test[feature], bins=30, alpha=0.5, color='orange', label='After Intervention')
            plt.title(f"Test Data: Distribution of '{feature}' Before and After Intervention")
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        # Compute confidence on test data after intervention
        self.compute_confidence_uncertainty_test()

        # Plot confidence before and after intervention on test data
        plt.figure(figsize=(10, 5))
        plt.hist(self.confidence_test_before_intervention, bins=30, alpha=0.5, label='Confidence Before Intervention')
        plt.hist(self.confidence_test, bins=30, alpha=0.5, color='green', label='Confidence After Intervention')
        plt.title(f"Test Data: Model Confidence Before and After Intervention onto {len(top_features)} features, RandomForest")
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        logging.info("Intervention complete. Observed changes in model confidence on test data.")

    def interpret(self, model):
        """
        Run the full interpretation process.
        """
        self.train_model(model=model)
        self.compute_confidence_uncertainty_train()
        self.estimate_feature_impact()
        self.perform_intervention()

# Usage Example
if __name__ == "__main__":
    # Example data loading function (replace with actual data)
    def load_data():
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize and run ModelInterpreter
    interpreter = ModelInterpreter(X_train, y_train, X_test, y_test)
    interpreter.interpret(RandomForestClassifier())
