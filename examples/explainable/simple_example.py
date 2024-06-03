import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

# Sample data
data = {
    'credit_score': np.random.randint(300, 850, size=500),
    'income': np.random.randint(30000, 100000, size=500),
    'loan_amount': np.random.randint(5000, 50000, size=500),
    'employment_status': np.random.binomial(1, 0.7, size=500),
    'loan_default': np.random.binomial(1, 0.2, size=500)
}
df = pd.DataFrame(data)

# Assume we have a machine learning model that predicts loan_default
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into train and test sets
X = df[['credit_score', 'income', 'loan_amount', 'employment_status']]
y = df['loan_default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Take a single prediction and its corresponding data point
test_instance = X_test.iloc[0]
print(f"Test Instance: {test_instance}")

# Get prediction for the test instance
predicted_default = model.predict([test_instance])[0]
print(f"Predicted Default: {predicted_default}")

# Create a causal model
causal_model = CausalModel(
    data=df,
    treatment=['credit_score', 'income', 'employment_status'],
    outcome='loan_default',
    graph="digraph { credit_score -> loan_default; income -> loan_default; employment_status -> loan_default; employment_status -> loan_amount; income -> loan_amount; credit_score -> loan_amount; }"
)

# Visualize the causal graph
causal_model.view_model(layout="dot")

# Identify causal effect
identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Estimate causal effect using Linear Regression Estimator
estimate = causal_model.estimate_effect(identified_estimand,
                                        method_name="backdoor.linear_regression")
print(estimate)

# Refute the estimate
refute = causal_model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
print(refute)

# Example interpretation
print(f"Causal Estimate: {estimate.value}")

# For example, if the estimate for credit_score is negative, it indicates that higher credit scores reduce the likelihood of defaulting on a loan.
if 'credit_score' in estimate.params:
    credit_score_effect = estimate.params['credit_score']
    if credit_score_effect < 0:
        print("Higher credit scores reduce the likelihood of defaulting on a loan.")
    else:
        print("Higher credit scores increase the likelihood of defaulting on a loan.")
