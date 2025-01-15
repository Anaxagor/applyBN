import numpy as np

class CausalFeatureSelector:
    def __init__(self, data=None, target=None, n_bins='auto'):
        """
        Initialize the object with optional data and target variables for causal feature selection.
        Optionally, the number of bins for discretization can also be specified.
        
        :param data: Feature data (matrix of features)
        :param target: Target variable (vector of target values)
        :param n_bins: Number of bins for discretization, or 'auto' for automatic selection
        """
        self.n_bins = n_bins
        self.selected_features = []  # List to store selected features
        self.other_features = np.array([])  # Array to store other features

        # If data and target are provided, discretize them
        if data is not None and target is not None:
            self.data = data
            self.target = target
            self.target_discretized = self.discretize_data_iqr(target)[0]
            self.data_discretized = np.array([self.discretize_data_iqr(data[:, i])[0] for i in range(data.shape[1])]).T

    def discretize_data_iqr(self, data):
        """
        Discretize the data using the interquartile range (IQR) rule for determining bin edges.
        
        :param data: Data to be discretized
        :return: Discretized data and bin edges
        """
        R = np.ptp(data)  # Range (max - min)
        iqr = np.subtract(*np.percentile(data, [75, 25]))  # Interquartile range

        if iqr == 0 or np.isclose(iqr, 0):
            iqr = 1e-8  # Avoid division by zero

        n = len(data)  # Number of observations

        # Calculate the number of bins automatically if 'auto' is selected
        if self.n_bins == 'auto':
            try:
                self.n_bins = max(2, int(np.ceil((R / (2 * iqr * n**(3/2))) * np.log2(n + 1))))
            except OverflowError:
                self.n_bins = 10  # Use a fixed number of bins in case of overflow

        # Bin edges
        bins = np.linspace(np.min(data), np.max(data), self.n_bins + 1)
        discretized_data = np.digitize(data, bins) - 1  # Return bin indices

        # Ensure discretized data values are less than len(bins) - 1
        discretized_data = np.clip(discretized_data, 0, len(bins) - 2)

        return discretized_data, bins

    def entropy(self, discretized_data):
        """
        Calculate the entropy of the discretized data.
        
        :param discretized_data: Discretized data for which entropy is calculated
        :return: Entropy of the data
        """
        if len(np.unique(discretized_data)) <= 1:
            return 0  # No variability in the data
        value_counts = np.unique(discretized_data, return_counts=True)[1]
        probabilities = value_counts / len(discretized_data)
        return -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))

    def conditional_entropy(self, X, Y):
        """
        Calculate the conditional entropy H(Y|X) using discretized data for X and Y.
        
        :param X: Discretized features
        :param Y: Discretized target variable
        :return: Conditional entropy H(Y|X)
        """
        X = np.asarray(X)  # Ensure X is an array
        Y = np.asarray(Y).flatten()  # Ensure Y is 1-dimensional

        if not np.issubdtype(X.dtype, np.integer) or not np.issubdtype(Y.dtype, np.integer):
            raise ValueError("X and Y must be discretized.")

        # Ensure X and Y have matching dimensions
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")

        cond_entropy = 0
        unique_x = np.unique(X)

        for x in unique_x:  # Loop through each unique value of X
            # Modify here to handle the case where X is 1D
            if X.ndim == 1:
                mask = (X == x)  # Compare directly
            else:
                mask = np.all(X == x, axis=1)  # Mask to extract all Y values corresponding to this value of X
            if mask.shape[0] != Y.shape[0]:
                raise ValueError("Mask size does not match Y size.")

            subset_entropy = self.entropy(Y[mask])  # Entropy of the subset of Y values
            cond_entropy += np.sum(mask) / len(X) * subset_entropy
        return cond_entropy

    def causal_effect(self, Xi, Y, other_features):
        """
        Calculate the causal effect of Xi on Y, accounting for other features.
        
        :param Xi: The feature for which the causal effect is calculated
        :param Y: The target variable
        :param other_features: Other features to control for
        :return: Causal effect of Xi on Y
        """
        Y_discretized = self.discretize_data_iqr(Y)[0]
        Xi_discretized = self.discretize_data_iqr(Xi)[0]

        # If there are other features, discretize them
        if other_features.size > 0:
            other_features_discretized = np.array([self.discretize_data_iqr(other_features[:, i])[0] for i in range(other_features.shape[1])]).T
        else:
            other_features_discretized = np.array([])

        # Ensure Xi and Y have matching dimensions
        if Xi.shape[0] != Y.shape[0]:
            raise ValueError("Xi and Y must have the same number of samples.")

        # Combine other features with Xi if needed
        if other_features_discretized.size > 0:
            if other_features_discretized.shape[0] != Y.shape[0]:
                raise ValueError("Other features and Y must have the same number of samples.")
            combined_features = np.c_[other_features_discretized, Xi_discretized]  # Combine features
            H_Y_given_other = self.conditional_entropy(other_features_discretized, Y_discretized)  # H(Y|other_features)
            H_Y_given_Xi_other = self.conditional_entropy(combined_features, Y_discretized)  # H(Y|Xi, other_features)
            return H_Y_given_other - H_Y_given_Xi_other  # Causal effect as the difference in entropies
        else:
            return self.entropy(Y_discretized) - self.conditional_entropy(Xi_discretized, Y_discretized)

    def causal_feature_selection(self, data, target):
        """
        Select features based on their causal effect on the target variable.
        
        :param data: The dataset containing features
        :param target: The target variable
        :return: List of selected feature indices
        """
        selected_features = []  # List to store selected features
        other_features = np.array([])  # Array to store already selected features

        target_discretized = self.discretize_data_iqr(target)[0]  # Discretize the target
        data_discretized = np.array([self.discretize_data_iqr(data[:, i])[0] for i in range(data.shape[1])]).T  # Discretize the features

        for i in range(data.shape[1]):  # Loop through each feature in the data
            feature = data[:, i]  # Current feature

            if len(feature) != len(target):  # Ensure the feature and target have the same size
                raise ValueError(f"Feature {i} and target have different sizes.")

            ce = self.causal_effect(feature, target, other_features)  # Calculate causal effect

            if ce > 0:  # If the causal effect is positive (significant)
                selected_features.append(i)  # Add the feature index to the list
                if other_features.size > 0:
                    other_features = np.c_[other_features, feature]  # Add the feature to other features
                else:
                    other_features = feature.reshape(-1, 1)  # Initialize other_features with the first feature

        return selected_features  # Return the list of selected feature indices
