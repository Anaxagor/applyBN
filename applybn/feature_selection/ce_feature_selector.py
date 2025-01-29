from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_X_y
import numpy as np

class CausalFeatureSelector(BaseEstimator, SelectorMixin):
    def __init__(self, n_bins: Union[int, str] = 'auto'):
        """
        Initialize the causal feature selector.

        Args:
            n_bins: Number of bins for discretization or 'auto' for automatic selection.
        """
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CausalFeatureSelector':
        """
        Fit the causal feature selector to the data.

        Args:
            X: Feature matrix (2D array).
            y: Target variable (1D array).

        Returns:
            self
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Compute the support mask for selected features
        self.support_ = self._select_features(X, y)
        return self

    def _select_features(self, data: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Core logic for selecting features based on causal influence.

        Args:
            data: Feature matrix.
            target: Target variable.

        Returns:
            Boolean mask of selected features.
        """
        # Discretize target and features
        target_discretized = self._discretize_data_iqr(target)[0]
        data_discretized = np.array([self._discretize_data_iqr(data[:, i])[0] for i in range(data.shape[1])]).T

        selected_mask = np.zeros(data.shape[1], dtype=bool)
        other_features = np.array([])

        for i in range(data.shape[1]):
            feature = data[:, i]
            # Compute causal effect of the current feature
            ce = self._causal_effect(feature, target, other_features)

            if ce > 0:  # If causal effect is significant
                selected_mask[i] = True
                if other_features.size > 0:
                    other_features = np.c_[other_features, feature]
                else:
                    other_features = feature.reshape(-1, 1)

        return selected_mask

    def _get_support_mask(self) -> np.ndarray:
        """
        Required by SelectorMixin to return the mask of selected features.

        Returns:
            Boolean mask of selected features.
        """
        return self.support_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the mask to the input data and return selected features.

        Args:
            X: Input feature matrix.

        Returns:
            Transformed feature matrix with only selected features.

        Raises:
            ValueError: If the number of features in X does not match the learned features.
        """
        # Validate the input array
        X = check_array(X)

        # Ensure the number of features matches the learned support mask
        if X.shape[1] != len(self.support_):
            raise ValueError("Number of features in X does not match the learned features.")
        
        return X[:, self.support_]

    def _discretize_data_iqr(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize data using the interquartile range (IQR) rule.

        Args:
            data: Data to discretize.

        Returns:
            Discretized data and bin edges.
        """
        # Compute range and IQR
        R = np.ptp(data)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        iqr = max(iqr, 1e-8)  # Avoid division by zero

        n = len(data)
        # Determine the number of bins
        n_bins = self.n_bins if self.n_bins != 'auto' else max(2, int(np.ceil((R / (2 * iqr * n ** (1/3))) * np.log2(n + 1))))

        bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
        discretized_data = np.digitize(data, bins) - 1
        discretized_data = np.clip(discretized_data, 0, len(bins) - 2)

        return discretized_data, bins

    def _causal_effect(self, Xi: np.ndarray, Y: np.ndarray, other_features: np.ndarray) -> float:
        """
        Compute the causal effect of Xi on Y, controlling for other features.

        Args:
            Xi: Feature for which the causal effect is calculated.
            Y: Target variable.
            other_features: Matrix of other features to control for.

        Returns:
            Causal effect of Xi on Y.
        """
        Y_discretized = self._discretize_data_iqr(Y)[0]
        Xi_discretized = self._discretize_data_iqr(Xi)[0]

        if other_features.size > 0:
            # Discretize other features if they exist
            other_features_discretized = np.array([self._discretize_data_iqr(other_features[:, i])[0] for i in range(other_features.shape[1])]).T
            combined_features = np.c_[other_features_discretized, Xi_discretized]
            H_Y_given_other = self._conditional_entropy(other_features_discretized, Y_discretized)
            H_Y_given_Xi_other = self._conditional_entropy(combined_features, Y_discretized)
            return H_Y_given_other - H_Y_given_Xi_other
        else:
            return self._entropy(Y_discretized) - self._conditional_entropy(Xi_discretized, Y_discretized)

    def _entropy(self, discretized_data: np.ndarray) -> float:
        """
        Compute the entropy of discretized data.

        Args:
            discretized_data: Discretized data.

        Returns:
            Entropy of the data.
        """
        value_counts = np.unique(discretized_data, return_counts=True)[1]
        probabilities = value_counts / len(discretized_data)
        return -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))

    def _conditional_entropy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the conditional entropy H(Y|X).

        Args:
            X: Discretized features.
            Y: Discretized target variable.

        Returns:
            Conditional entropy H(Y|X).
        """
        unique_x = np.unique(X, axis=0)
        cond_entropy = 0

        for x in unique_x:
            # Mask for rows where X equals the current unique value
            if X.ndim == 1:
                mask = (X == x)  # Simple comparison for 1D array
            else:
                mask = np.all(X == x, axis=1)
            subset_entropy = self._entropy(Y[mask])
            cond_entropy += np.sum(mask) / len(X) * subset_entropy

        return cond_entropy
