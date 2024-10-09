import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from torch.nn.utils.prune import BasePruningMethod

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

class CausalPruningMethod(BasePruningMethod):
    def __init__(self, model, train_loader, test_loader, percentage=1.0):
        """
        Initialize the CausalPruningMethod class.

        Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        percentage (float): The percentage of filters to prune.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.percentage = percentage
        self.filter_responses = {}
        self.transformations = {}
        self.regressors = {}
        self.conv_layers = self.get_conv_layers()
        self.run_scm()

    def get_conv_layers(self):
        """Retrieve all convolutional layers from the model."""
        conv_layers = []
        for layer in self.model.children():
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)
        return conv_layers

    def extract_filter_responses(self):
        """Extract filter responses from each convolutional layer in the model."""
        hooks = []

        def hook_fn(module, input, output):
            if module not in self.filter_responses:
                self.filter_responses[module] = []
            self.filter_responses[module].append(output.cpu().numpy())

        # Register hooks for each convolutional layer
        for layer in self.conv_layers:
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
        layer_index = self.conv_layers.index(layer)
        if layer_index > 0:
            parents.append(self.conv_layers[layer_index - 1])
        return parents

    def compute_mask(self, t, default_mask):
        """
        Compute the mask for pruning based on the importance of the filters.

        Parameters:
        t (torch.Tensor): The tensor to be pruned.
        default_mask (torch.Tensor): The default mask.

        Returns:
        torch.Tensor: The mask to apply for pruning.
        """
        layer = self.current_layer
        importance = self.predict_importance(layer)
        num_filters_to_zero = int(len(importance) * self.percentage / 100)
        least_important_filters = np.argsort(importance)[:num_filters_to_zero]
        mask = default_mask.clone()
        mask[least_important_filters] = 0
        return mask

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

    def run_scm(self):
        """Run the structural causal model to extract filter responses and learn structural equations."""
        self.extract_filter_responses()
        self.apply_transformation()
        self.learn_structural_equations()

    def prune(self, model):
        """
        Prune the model based on the learned causal relationships.

        Parameters:
        model (nn.Module): The model to prune.
        """
        for layer in self.conv_layers:
            self.current_layer = layer
            prune_custom_from_mask = torch.nn.utils.prune.custom_from_mask
            prune_custom_from_mask(layer, name="weight", mask=self.compute_mask(layer.weight, torch.ones_like(layer.weight)))
            if layer.bias is not None:
                prune_custom_from_mask(layer, name="bias", mask=self.compute_mask(layer.bias, torch.ones_like(layer.bias)))


def foobar_unstructured(module, name):
    """
    Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module
    """
    CausalPruningMethod.apply(module, name)
    return module



# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224 as required by ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalize using ImageNet statistics
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load pre-trained ResNet-18
model = models.shufflenet_v2_x0_5(pretrained=True)

# Modify the final layer to match the number of classes in CIFAR-10 (10 classes)
model.fc = nn.Linear(model.fc.in_features, 10)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fine-tuning settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Fine-tuning loop
for epoch in range(2):  # Fine-tuning for 5 epochs for demonstration
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Fine-Tuning')


def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


print(f'Accuracy before pruning: {test_accuracy(model, test_loader):.2f}%')

# Assuming the CausalPruningMethod and foobar_unstructured are defined as before

# Initialize the pruning method with the pre-trained model and data loaders
pruning_method = CausalPruningMethod(model, train_loader, test_loader)
pruning_method.run_scm()  # Run the SCM to determine filter importance


# Prune the model using the custom method
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        pruning_method.apply(module, name="weight")

print('Finished Pruning')

print(f'Accuracy after pruning: {test_accuracy(model, test_loader):.2f}%')
