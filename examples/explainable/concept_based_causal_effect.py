import numpy as np
import pandas as pd
import logging

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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


def select_features_for_concept(
    concept_data, other_data, features, original_data, lambda_reg=0.1
):
    """Selects features for a concept and extracts value ranges or categories."""
    selected_features = {}
    for feature in features:
        X_i_feature = concept_data[feature]
        X_minus_i_feature = other_data[feature]
        # For numerical features, calculate the mean difference
        if feature in original_data.columns and original_data[feature].dtype != object:
            mean_diff = abs(X_i_feature.mean() - X_minus_i_feature.mean())
            var_within = X_i_feature.var()
            score = mean_diff - lambda_reg * var_within
            if score > 0:
                # Get value range from original data
                original_feature = feature
                indices = concept_data.index
                X_i_orig_feature = original_data.loc[indices, original_feature]
                value_range = (X_i_orig_feature.min(), X_i_orig_feature.max())
                selected_features[original_feature] = {
                    "type": "numeric",
                    "range": value_range,
                }
        else:
            # For one-hot encoded categorical features
            proportion_in_concept = X_i_feature.mean()
            proportion_in_others = X_minus_i_feature.mean()
            proportion_diff = proportion_in_concept - proportion_in_others
            score = abs(proportion_diff) - lambda_reg
            if score > 0:
                # Map back to original feature and category
                if "_" in feature:
                    original_feature = "_".join(feature.split("_")[:-1])
                    category = feature.split("_")[-1]
                    selected_features.setdefault(
                        original_feature, {"type": "categorical", "categories": []}
                    )
                    selected_features[original_feature]["categories"].append(category)
    return selected_features


def extract_concept_meanings(D, cluster_concepts, original_data):
    """Extracts the meanings of the concepts."""
    selected_features_per_concept = {}
    features = D.drop(columns="index").columns.tolist()

    for idx, concept in enumerate(cluster_concepts):
        cluster_indices = concept["cluster_indices"]
        concept_data = D.set_index("index").loc[cluster_indices]
        other_indices = D.index.difference(concept_data.index)
        other_data = D.loc[other_indices]
        selected_features = select_features_for_concept(
            concept_data, other_data, features, original_data
        )
        selected_features_per_concept[f"Concept_{idx}"] = selected_features
        logger.info(f"\nConcept_{idx} selected features and values:")
        for feature, details in selected_features.items():
            if details["type"] == "numeric":
                logger.info(f"  {feature}: range {details['range']}")
            else:
                logger.info(f"  {feature}: categories {details['categories']}")
    return selected_features_per_concept


def estimate_causal_effects(D_c):
    """Estimates the causal effect of each concept on L_f using logistic regression."""
    import statsmodels.api as sm

    effects = {}
    outcome = "L_f"
    for concept in D_c.columns:
        if concept != outcome:
            # Prepare data
            X = D_c[[concept]].copy()
            # Control for other concepts
            other_concepts = [
                col for col in D_c.columns if col not in [concept, outcome]
            ]
            if other_concepts:
                X[other_concepts] = D_c[other_concepts]
            X = sm.add_constant(X)
            y = D_c[outcome]
            # Fit logistic regression
            model = sm.Logit(y, X).fit(disp=0)
            # Extract the coefficient for the concept
            coef = model.params[concept]
            effects[concept] = coef
            logger.info(
                f"{concept}: Estimated Coefficient (Causal Effect) = {coef:.4f}"
            )
    return effects


def main():
    """Main function to run the CMIC algorithm."""
    X, y, original_X = load_and_preprocess_data()

    # D (discovery dataset) and N (natural dataset)
    D, N = train_test_split(X, test_size=0.3, random_state=42, shuffle=False)
    D.reset_index(drop=False, inplace=True)
    N.reset_index(drop=False, inplace=True)
    original_X.reset_index(drop=True, inplace=True)

    # Extract concepts
    cluster_concepts = extract_concepts(D, N)

    # Generate concept space
    A = generate_concept_space(X, cluster_concepts)

    predictive_model = RandomForestClassifier(n_estimators=100, random_state=42)
    predictive_model.fit(X, y)
    L_f = predictive_model.predict(X)

    D_c = A.copy()
    D_c["L_f"] = L_f

    # Extract concept meanings
    extract_concept_meanings(
        D, cluster_concepts, original_data=original_X
    )

    # Estimate causal effects
    effects = estimate_causal_effects(D_c)

    sorted_effects = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
    logger.info(
        "\nRanking of Concepts by Estimated Coefficient (Causal Effect) on L_f:"
    )
    for concept, effect in sorted_effects:
        logger.info(f"{concept}: Causal Effect = {effect:.4f}")


if __name__ == "__main__":
    main()
