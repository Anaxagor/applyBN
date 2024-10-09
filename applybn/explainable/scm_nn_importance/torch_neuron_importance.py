import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression


class NeuronImportanceSCM:
    def __init__(self, model, train_loader, test_loader):
        """
        Initialize the NeuronImportanceSCM class.

        Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.neuron_responses = {}
        self.transformations = {}
        self.regressors = {}
        self.fc_layers = self.get_neuron_layers()

    def get_neuron_layers(self):
        """Retrieve all fully connected layers from the model."""
        neuron_layers = []
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                neuron_layers.append(layer)
        return neuron_layers

    def extract_neuron_responses(self):
        """Extract neuron responses from each fully connected layer in the model."""
        hooks = []

        def hook_fn(module, input, output):
            if module not in self.neuron_responses:
                self.neuron_responses[module] = []
            self.neuron_responses[module].append(output.cpu().numpy())

        # Register hooks for each fully connected layer
        for layer in self.fc_layers:
            hooks.append(layer.register_forward_hook(hook_fn))

        self.model.eval()
        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                _ = self.model(data)  # Forward pass

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Concatenate neuron responses
        for layer in self.neuron_responses:
            self.neuron_responses[layer] = np.concatenate(
                self.neuron_responses[layer], axis=0
            )

    def apply_transformation(self):
        """Apply transformation to the neuron responses using the Frobenius norm."""
        for layer, responses in self.neuron_responses.items():
            if responses.size == 0:
                print(f"Warning: No responses found for layer: {layer}")
            self.transformations[layer] = np.linalg.norm(responses, axis=0)

    def learn_structural_equations(self):
        """Learn structural equations using linear regression for each layer."""
        for layer, transformed in self.transformations.items():
            parents = self.get_parents(layer)
            if not parents:
                print(
                    f"Skipping structural equation learning for the first layer: {layer}"
                )
                continue  # Skip the first layer as it has no parents

            # Print shapes of transformed data
            print(f"Layer: {layer}, Transformed shape: {transformed.shape}")

            parent_transformed = []
            for parent in parents:
                parent_data = self.transformations[parent]
                print(
                    f"Parent Layer: {parent}, Parent Transformed shape: {parent_data.shape}"
                )

                if parent_data.ndim == 1:
                    parent_data = parent_data[:, np.newaxis]
                if parent_data.shape[0] > transformed.shape[0]:
                    parent_data = parent_data[: transformed.shape[0], :]
                parent_transformed.append(parent_data)

            X = np.concatenate(parent_transformed, axis=1)
            y = transformed if transformed.ndim == 1 else transformed.mean(axis=0)

            # Ensure X and y have the same number of samples
            if X.shape[0] != y.shape[0]:
                min_samples = min(X.shape[0], y.shape[0])
                X = X[:min_samples]
                y = y[:min_samples]

            print(f"X shape: {X.shape}, y shape: {y.shape}")

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
        layer_index = self.fc_layers.index(layer)
        if layer_index > 0:
            parents.append(self.fc_layers[layer_index - 1])
        return parents

    def predict_neuron_importance(self, layer):
        """
        Predict the importance of neurons in a given layer.

        Parameters:
        layer (nn.Module): The layer for which to predict neuron importance.

        Returns:
        numpy.ndarray: Array of neuron importance values.
        """
        regressor = self.regressors.get(layer)
        if regressor is None:
            raise ValueError(
                f"No structural equation learned for layer: {layer}. Importance cannot be predicted."
            )

        importance = np.zeros(self.transformations[layer].shape[0])
        for i in range(len(importance)):
            importance[i] = (
                regressor.coef_[i % len(regressor.coef_)]
                * self.transformations[layer][i]
            )
        return importance

    def zero_least_important_neurons(self, layer, percentage=1):
        """
        Zero out the least important neurons in a given layer.

        Parameters:
        layer (nn.Module): The layer in which to zero out neurons.
        percentage (float): The percentage of neurons to zero out.
        """
        importance = self.predict_neuron_importance(layer)
        num_neurons_to_zero = int(len(importance) * percentage / 100)
        least_important_neurons = np.argsort(importance)[:num_neurons_to_zero]
        with torch.no_grad():
            layer.weight[:, least_important_neurons] = 0

    def zero_least_important_neurons_across_layers(self, percentage=1):
        """
        Zero out the least important neurons across all layers.

        Parameters:
        percentage (float): The percentage of neurons to zero out across the entire model.
        """
        all_neurons = []
        total_neurons = 0

        # Collect neuron importances across all layers
        for layer in self.fc_layers:
            if (
                layer in self.regressors
            ):  # Skip layers without learned structural equations
                importance = self.predict_neuron_importance(layer)
                total_neurons += len(importance)
                all_neurons.extend(
                    [(layer, i, importance[i]) for i in range(len(importance))]
                )

        # Determine how many neurons to prune
        num_neurons_to_zero = int(total_neurons * percentage / 100)

        # Sort neurons by importance
        all_neurons.sort(key=lambda x: x[2])  # Sort by the importance value

        # Prune the least important neurons
        neurons_to_zero = all_neurons[:num_neurons_to_zero]

        with torch.no_grad():
            for layer, neuron_index, _ in neurons_to_zero:
                print(
                    f"Pruning neuron {neuron_index} in layer {layer} with importance {_}"
                )
                layer.weight[:, neuron_index] = 0
                if layer.bias is not None and layer.bias.size(0) > neuron_index:
                    layer.bias[neuron_index] = 0

    def zero_random_neurons_across_layers(self, percentage=1):
        """
        Zero out random neurons across all layers.

        Parameters:
        percentage (float): The percentage of neurons to zero out across the entire model.
        """
        total_neurons = sum(layer.weight.size(1) for layer in self.fc_layers)
        num_neurons_to_zero = int(total_neurons * percentage / 100)

        all_neurons = []
        for layer in self.fc_layers:
            all_neurons.extend([(layer, i) for i in range(layer.weight.size(1))])

        random_neurons = np.random.choice(
            len(all_neurons), num_neurons_to_zero, replace=False
        )
        neurons_to_zero = [all_neurons[i] for i in random_neurons]

        with torch.no_grad():
            for layer, neuron_index in neurons_to_zero:
                print(f"Randomly pruning neuron {neuron_index} in layer {layer}")
                layer.weight[:, neuron_index] = 0
                if layer.bias is not None and layer.bias.size(0) > neuron_index:
                    layer.bias[neuron_index] = 0

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
        """Run the structural causal model to extract neuron responses and learn structural equations."""
        self.extract_neuron_responses()
        self.apply_transformation()
        self.learn_structural_equations()
