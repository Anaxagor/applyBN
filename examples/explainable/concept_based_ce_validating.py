import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from econml.dml import LinearDML, CausalForestDML

from applybn.explainable.causal_explain.data_iq import DataIQSKLearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data = pd.read_csv(url, names=column_names, header=None, na_values=" ?")

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    X = data.drop("income", axis=1).reset_index(drop=True)
    y = (
        data["income"]
        .apply(lambda x: 1 if x.strip() == ">50K" else 0)
        .reset_index(drop=True)
    )

    categorical_cols = X.select_dtypes(include=["object"]).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_encoded = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
    )
    X_numeric = X.select_dtypes(exclude=["object"]).reset_index(drop=True)
    X_processed = pd.concat(
        [X_numeric.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1
    )

    scaler = StandardScaler()
    numeric_cols = X_numeric.columns
    X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])

    X_processed.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)

    return X_processed, y, X


def calculate_confidence_uncertainty(X, y, clf):
    """Calculates model confidence and aleatoric uncertainty using Data-IQ."""
    data_iq = DataIQSKLearn(X=X, y=y)
    data_iq.on_epoch_end(clf=clf, iteration=10)
    confidence = data_iq.confidence
    aleatoric_uncertainty = data_iq.aleatoric
    return confidence, aleatoric_uncertainty


def perform_clustering(D, num_clusters):
    """Performs KMeans clustering on dataset D."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(D)
    return clusters


def evaluate_discriminability(D, N, clusters, auc_threshold, k_min_cluster_size):
    """Evaluates discriminability of clusters and returns discriminative clusters."""
    discriminative_clusters = []
    cluster_labels = np.unique(clusters)
    for cluster_label in cluster_labels:
        cluster_indices = np.where(clusters == cluster_label)[0]
        if len(cluster_indices) >= k_min_cluster_size:
            # Prepare data for SVM classifier
            X_cluster = D.iloc[cluster_indices]
            y_cluster = np.ones(len(cluster_indices))
            X_N = N.drop(columns="index")  # Negative class is the natural dataset N
            y_N = np.zeros(len(X_N))
            # Combine datasets
            X_train = pd.concat([X_cluster.drop(columns="index"), X_N], axis=0)
            y_train = np.concatenate([y_cluster, y_N])
            # Split into training and validation sets
            X_train_svm, X_val_svm, y_train_svm, y_val_svm = train_test_split(
                X_train, y_train, test_size=0.3, random_state=42
            )
            # Train SVM
            S_i = SVC(kernel="linear", probability=True, random_state=42)
            S_i.fit(X_train_svm, y_train_svm)
            # Evaluate discriminability using AUC
            y_scores = S_i.decision_function(X_val_svm)
            auc_score = roc_auc_score(y_val_svm, y_scores)
            logger.info(f"Cluster {cluster_label}: AUC = {auc_score:.4f}")
            if auc_score > auc_threshold:
                # Discriminative cluster found
                discriminative_clusters.append(
                    {
                        "classifier": S_i,
                        "cluster_label": cluster_label,
                        "cluster_indices": D.iloc[cluster_indices]["index"].values,
                        "auc_score": auc_score,
                    }
                )
    return discriminative_clusters


def extract_concepts(
    D, N, auc_threshold=0.7, k_min_cluster_size=100, max_clusters=10, max_iterations=10
):
    """Extracts concepts from dataset D."""
    svm_classifiers = []
    cluster_concepts = []
    Nc = 0  # Number of concepts
    no_improvement_counter = 0

    num_clusters = 9
    iteration = 0

    while num_clusters <= max_clusters and iteration < max_iterations:
        iteration += 1
        logger.info(f"\nIteration {iteration}: Clustering with {num_clusters} clusters")
        clusters = perform_clustering(D.drop(columns="index"), num_clusters)
        discriminative_clusters = evaluate_discriminability(
            D, N, clusters, auc_threshold, k_min_cluster_size
        )

        if discriminative_clusters:
            for concept in discriminative_clusters:
                svm_classifiers.append(concept["classifier"])
                cluster_concepts.append(concept)
                Nc += 1
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= 3:
            logger.info("No significant improvement in discriminability.")
            break

        num_clusters += 1

    return cluster_concepts


def generate_concept_space(X, cluster_concepts):
    """Generates the concept space A."""
    A = pd.DataFrame(index=X.index)
    for idx, concept in enumerate(cluster_concepts):
        classifier = concept["classifier"]
        A_i_scores = classifier.decision_function(X)
        A[f"Concept_{idx}"] = (A_i_scores > 0).astype(int)
    return A


def main():
    """Main function to compare distributions of predicted and actual confidence and uncertainty."""
    X_processed, y, original_X = load_and_preprocess_data()

    # Split data into training and validation sets
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Reset indices to ensure they are zero-based and contiguous
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_validation.reset_index(drop=True, inplace=True)
    y_validation.reset_index(drop=True, inplace=True)

    # From training set, split into D and N for concept extraction
    D, N = train_test_split(X_train, test_size=0.3, random_state=42, shuffle=False)
    D.reset_index(drop=False, inplace=True)
    N.reset_index(drop=False, inplace=True)
    original_X.reset_index(drop=True, inplace=True)

    # Extract concepts from D and N
    cluster_concepts = extract_concepts(D, N)

    # Generate concept space for training and validation sets
    A_train = generate_concept_space(X_train, cluster_concepts)
    A_validation = generate_concept_space(X_validation, cluster_concepts)

    # Train predictive model on training set
    predictive_model = RandomForestClassifier(n_estimators=100, random_state=42)
    predictive_model.fit(X_train, y_train)

    # Calculate confidence and uncertainty on validation set (actual values)
    confidence_validation, uncertainty_validation = calculate_confidence_uncertainty(
        X_validation, y_validation, predictive_model
    )
    # Convert to pandas Series with appropriate indices
    confidence_validation = pd.Series(confidence_validation, index=X_validation.index)
    uncertainty_validation = pd.Series(uncertainty_validation, index=X_validation.index)

    # Compute average confidence and uncertainty per concept on training set
    # (we'll use these to predict values on validation set)
    # First, calculate confidence and uncertainty on training set
    confidence_train, uncertainty_train = calculate_confidence_uncertainty(
        X_train, y_train, predictive_model
    )
    confidence_train = pd.Series(confidence_train, index=X_train.index)
    uncertainty_train = pd.Series(uncertainty_train, index=X_train.index)

    concepts = A_train.columns.tolist()
    concept_confidence = {}
    concept_uncertainty = {}

    for concept in concepts:
        indices = A_train[A_train[concept] == 1].index
        if len(indices) > 0:
            avg_conf = confidence_train.loc[indices].mean()
            avg_uncert = uncertainty_train.loc[indices].mean()
        else:
            avg_conf = confidence_train.mean()
            avg_uncert = uncertainty_train.mean()
        concept_confidence[concept] = avg_conf
        concept_uncertainty[concept] = avg_uncert

    # Predict confidence and uncertainty for validation set based on concepts
    predicted_confidence = []
    predicted_uncertainty = []

    for idx in A_validation.index:
        concepts_active = A_validation.loc[idx]
        active_concepts = concepts_active[concepts_active == 1].index.tolist()
        if len(active_concepts) == 0:
            # Assign overall average from training set
            avg_conf = confidence_train.mean()
            avg_uncert = uncertainty_train.mean()
        else:
            # Sum the concept confidences and uncertainties
            conf_sum = sum([concept_confidence[concept] for concept in active_concepts])
            uncert_sum = sum([concept_uncertainty[concept] for concept in active_concepts])
            # Normalize by the number of active concepts
            avg_conf = conf_sum / len(active_concepts)
            avg_uncert = uncert_sum / len(active_concepts)
        predicted_confidence.append(avg_conf)
        predicted_uncertainty.append(avg_uncert)

    # Convert predicted lists to pandas Series with the same index as A_validation
    predicted_confidence = pd.Series(predicted_confidence, index=A_validation.index)
    predicted_uncertainty = pd.Series(predicted_uncertainty, index=A_validation.index)

    # Create a DataFrame to hold predicted and actual values with proper indices
    results_df = pd.DataFrame({
        'predicted_confidence': predicted_confidence,
        'actual_confidence': confidence_validation,
        'predicted_uncertainty': predicted_uncertainty,
        'actual_uncertainty': uncertainty_validation
    })

    # Drop any rows with missing values (if any)
    results_df.dropna(inplace=True)

    # Compare distributions of predicted and actual confidence on validation set
    plt.figure(figsize=(12, 6))
    sns.kdeplot(results_df['predicted_confidence'], label='Predicted Confidence', shade=True)
    sns.kdeplot(results_df['actual_confidence'], label='Actual Confidence', shade=True)
    plt.title('Distributions of Predicted vs. Actual Confidence on Validation Set')
    plt.xlabel('Confidence')
    plt.legend()
    plt.show()

    # Compare distributions of predicted and actual uncertainty on validation set
    plt.figure(figsize=(12, 6))
    sns.kdeplot(results_df['predicted_uncertainty'], label='Predicted Uncertainty', shade=True)
    sns.kdeplot(results_df['actual_uncertainty'], label='Actual Uncertainty', shade=True)
    plt.title('Distributions of Predicted vs. Actual Uncertainty on Validation Set')
    plt.xlabel('Uncertainty')
    plt.legend()
    plt.show()

    # Perform statistical tests to compare distributions
    from scipy.stats import ks_2samp

    ks_conf = ks_2samp(results_df['predicted_confidence'], results_df['actual_confidence'])
    ks_uncert = ks_2samp(results_df['predicted_uncertainty'], results_df['actual_uncertainty'])

    logger.info(
        f"Confidence KS Statistic: {ks_conf.statistic:.4f}, p-value: {ks_conf.pvalue:.4f}"
    )
    logger.info(
        f"Uncertainty KS Statistic: {ks_uncert.statistic:.4f}, p-value: {ks_uncert.pvalue:.4f}"
    )

    # Optionally, compute and print correlation coefficients
    corr_confidence = results_df['predicted_confidence'].corr(results_df['actual_confidence'])
    corr_uncertainty = results_df['predicted_uncertainty'].corr(results_df['actual_uncertainty'])
    logger.info(f"Correlation between predicted and actual confidence: {corr_confidence:.4f}")
    logger.info(f"Correlation between predicted and actual uncertainty: {corr_uncertainty:.4f}")

    # You can also plot box plots for a different visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df[['predicted_confidence', 'actual_confidence']])
    plt.title('Box Plot of Predicted vs. Actual Confidence on Validation Set')
    plt.ylabel('Confidence')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=results_df[['predicted_uncertainty', 'actual_uncertainty']])
    plt.title('Box Plot of Predicted vs. Actual Uncertainty on Validation Set')
    plt.ylabel('Uncertainty')
    plt.show()


if __name__ == "__main__":
    main()
