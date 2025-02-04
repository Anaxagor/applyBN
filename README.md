# applybn

applybn is an open-source multi-purpose framework based on Bayesian networks and Causal networks.
The main idea is to implement the main functions of data analysis based on understandable and interpretable algorithms of Bayesian networks and causal models.
![image](https://github.com/user-attachments/assets/996f8e5a-1742-4849-a64f-58b97a4cf17d)


**The repository is currently work-in-progress**

## Key Features
### 1. Anomaly Detection in time-series and tabular data
#### **Local Outlier Factor (LOF) Based Anomaly Detection**
   - **Description**: The LOF algorithm computes the local density deviation of a given data point compared to its neighbors, helping identify points that significantly deviate in terms of density.
   - **How It Works**: On each iteration, `n` random columns are selected, and LOF scores are calculated for each observation in the dataset. These scores are then normalized between 0 and 1.
   
#### **Bayesian Network-Based Anomaly Detection**
   - **Description**: This method leverages the trained Bayesian network to evaluate the conditional distributions of nodes based on their parents, detecting anomalies in the data based on their deviation from expected conditional probabilities.
   - **How It Works**: During each iteration, a random node and its parents are selected, and an anomaly score is computed based on the conditional distribution of values at that node for each observation. These scores are normalized between 0 and 1, providing a clear metric of anomaly likelihood.

#### **Combined Anomaly Scoring**
   - **Description**: The final anomaly score for each observation is calculated by combining the normalized LOF scores and Bayesian network scores. This combination allows for both density-based and dependency-based anomalies to be detected simultaneously.
   - **Formula**: The combined anomaly score for each observation is computed using a pre-defined formula (3.2), which balances both types of anomaly scores.

### 2. **Synthetic Data Generation**
   - **Class Imbalance Handling**: The framework includes methods for generating synthetic training data when class imbalance is detected in the dataset. Using hybrid Bayesian networks (with Gaussian mixture models), it generates balanced synthetic data, improving model training outcomes.
   - **Synthetic Test Data Generation**: applybn can generate synthetic test datasets, ensuring the generated samples are representative enough for proper model evaluation. This feature helps address issues when there is a lack of sufficient real test data, using a unique condition to ensure that ranking of model errors on synthetic and real data remains consistent.


### 3. **Feature Selection module**
  - **Feature Selection** for label prediction.
  - **Feature Generation** for label prediction enchancement.

### 4. **Explainable Module**
#### **Causal Analysis for Machine Learning Models**
   - **Analyzing Model Components**: A structural causal model (SCM) is built to analyze deep learning models, allowing for the pruning of unimportant parts (e.g., filters in CNNs) by evaluating their causal importance.
   - **Explaining Data Impact on Predictions**: applybn allows for causal inference between features and the model’s confidence scores. By calculating the **Average Causal Effect (ACE)**, it helps identify which features significantly influence model uncertainty, providing valuable insights for improving or debugging models.

## Work-in-Progress

applybn is actively under development, with key features currently being tested in dedicated branches. Contributions to these features are welcome:

- [Anomaly Detection Module](https://github.com/Anaxagor/applyBN/tree/anomaly-detection-module)
- [Data Generation Module](https://github.com/Anaxagor/applyBN/tree/data-generation-module)
- [Feature Selection Module](https://github.com/Anaxagor/applyBN/tree/feature-selection-module)
- [Explainable Module](https://github.com/Anaxagor/applyBN/tree/explainable-module)

## Installation

To get started with applybn, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Anaxagor/applybn.git
cd applybn
pip install -r requirements.txt
```

## Usage Example

API is WIP.

## Contributing

Contributions to applybn are welcome! If you’re interested in improving any of the features or testing new branches, please see the `CONTRIBUTING.md` (WIP) file for details.

## License

applybn is distributed under the MIT License. See the `LICENSE` file for more information.

For additional documentation and technical details, visit our [documentation](https://anaxagor.github.io/applybn/).
