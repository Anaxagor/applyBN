import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression


class FilterImportanceSCM:
    def __init__(self, model, train_loader, test_loader):
        """
        Initialize the FilterImportanceSCM class.

        Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.filter_responses = {}
        self.transformations = {}
        self.regressors = {}

    def extract_filter_responses(self):
        """Extract filter responses from each convolutional layer in the model."""
        hooks = []

        def hook_fn(module, input, output):
            if module not in self.filter_responses:
                self.filter_responses[module] = []
            self.filter_responses[module].append(output.cpu().numpy())

        # Register hooks for each convolutional layer
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(hook_fn))

        self.model.eval()
        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                _ = self.model(data)  # Forward pass

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate filter responses
        for layer in self.filter_responses:
            self.filter_responses[layer] = np.concatenate(
                self.filter_responses[layer], axis=0
            )

    def apply_transformation(self):
        """Apply transformation to the filter responses using the Frobenius norm."""
        for layer, responses in self.filter_responses.items():
            self.transformations[layer] = np.linalg.norm(responses, axis=(2, 3))

    def learn_structural_equations(self):
        """Learn structural equations using linear regression for each layer."""
        for layer, transformed in self.transformations.items():
            parents = self.get_parents(layer)
            if not parents:
                continue
            X = np.concatenate(
                [self.transformations[parent] for parent in parents], axis=1
            )
            y = transformed.mean(axis=1)
            regressor = LinearRegression().fit(X, y)
            self.regressors[layer] = regressor

    def get_parents(self, layer):
        """
        Return the list of parent layers based on the DAG structure of the model.

        Parameters:
        layer (nn.Module): The layer for which to find the parents.

        Returns:
        list: List of parent layers.
        """
        parents = []
        if layer == self.model.conv2:
            parents.append(self.model.conv1)
        elif layer == self.model.conv3:
            parents.append(self.model.conv2)
        return parents

    def predict_importance(self, layer):
        """
        Predict the importance of filters in a given layer.

        Parameters:
        layer (nn.Module): The layer for which to predict filter importance.

        Returns:
        numpy.ndarray: Array of filter importance values.
        """
        regressor = self.regressors.get(layer)
        if regressor is None:
            return None
        importance = np.zeros(self.transformations[layer].shape[1])
        for i in range(len(importance)):
            importance[i] = (
                regressor.coef_[i % len(regressor.coef_)]
                * self.transformations[layer][:, i].mean()
            )
        return importance

    def zero_least_important_filters(self, layer, percentage=1):
        """
        Zero out the least important filters in a given layer.

        Parameters:
        layer (nn.Module): The layer in which to zero out filters.
        percentage (float): The percentage of filters to zero out.
        """
        importance = self.predict_importance(layer)
        num_filters_to_zero = int(len(importance) * percentage / 100)
        least_important_filters = np.argsort(importance)[:num_filters_to_zero]
        with torch.no_grad():
            layer.weight[least_important_filters] = 0
            if layer.bias is not None:
                layer.bias[least_important_filters] = 0

    def zero_random_filters(self, layer, percentage=1):
        """
        Zero out random filters in a given layer.

        Parameters:
        layer (nn.Module): The layer in which to zero out filters.
        percentage (float): The percentage of filters to zero out.
        """
        num_filters_to_zero = int(layer.weight.size(0) * percentage / 100)
        random_filters = np.random.choice(
            layer.weight.size(0), num_filters_to_zero, replace=False
        )
        with torch.no_grad():
            layer.weight[random_filters] = 0
            if layer.bias is not None:
                layer.bias[random_filters] = 0

    def measure_accuracy(self):
        """
        Measure the accuracy of the model on the test data.

        Returns:
        float: The accuracy of the model.
        """
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct / total

    def run_scm(self):
        """Run the structural causal model to extract filter responses and learn structural equations."""
        self.extract_filter_responses()
        self.apply_transformation()
        self.learn_structural_equations()
