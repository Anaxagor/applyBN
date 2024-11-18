import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import copy
import random
import logging
import time

from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),  # Normalization parameters for ImageNet
                         (0.229, 0.224, 0.225))
])
logger.info('Transformations defined.')

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)
logger.info('CIFAR-10 dataset loaded.')

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=2)
logger.info('Data loaders created.')

# Load pre-trained ResNet18 model
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
logger.info('Pre-trained ResNet18 model loaded.')

# Modify the last fully connected layer to output 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
logger.info('Model modified for CIFAR-10.')

# Move the model to the device
model = model.to(device)
logger.info('Model moved to device.')

# Function to train the model
def train_model(model, train_loader, num_epochs=5):
    logger.info('Starting model training...')
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_time = time.time() - epoch_start_time
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%, Time: {epoch_time:.2f}s')

    total_time = time.time() - start_time
    logger.info(f'Finished Training in {total_time:.2f}s')

# Fine-tune the model
train_model(model, train_loader, num_epochs=5)

# Define the CNN_SCM class to construct the SCM over the CNN
class CNN_SCM:
    def __init__(self, model):
        self.model = model
        self.conv_layers = []
        self._extract_conv_layers()
        self.dag = self._build_dag()
        logger.info('CNN_SCM initialized.')

    def _extract_conv_layers(self):
        # Recursively extract convolutional layers
        self.conv_layers = []
        self.layer_names = []

        def recursive_extract(module, name_prefix=''):
            for name, layer in module.named_children():
                full_name = f"{name_prefix}.{name}" if name_prefix else name
                if isinstance(layer, nn.Conv2d):
                    self.conv_layers.append((full_name, layer))
                    self.layer_names.append(full_name)
                else:
                    recursive_extract(layer, full_name)

        recursive_extract(self.model)
        logger.info(f'Extracted {len(self.conv_layers)} convolutional layers.')

    def _build_dag(self):
        # Build a DAG where each node represents a filter
        dag = {}
        for idx, (name, layer) in enumerate(self.conv_layers):
            num_filters = layer.out_channels
            dag[idx] = {
                'name': name,
                'layer': layer,
                'filters': list(range(num_filters)),
                'parents': [idx - 1] if idx > 0 else [],
                'children': [idx + 1] if idx < len(self.conv_layers) - 1 else []
            }
        return dag

# Instantiate the SCM over the CNN
cnn_scm = CNN_SCM(model)

# Define the transformation function (Frobenius Norm)
def frobenius_norm(tensor):
    # Compute the Frobenius norm over spatial dimensions
    return torch.norm(tensor.view(tensor.size(0), tensor.size(1), -1), dim=2)

# Helper function to get submodule by name
def get_submodule(model, target):
    names = target.split('.')
    submodule = model
    for name in names:
        if name.isdigit():
            submodule = submodule[int(name)]
        else:
            submodule = getattr(submodule, name)
    return submodule

# Define the StructuralEquations class to learn the structural equations and compute filter importances
class StructuralEquations:
    def __init__(self, cnn_scm):
        self.cnn_scm = cnn_scm
        self.structural_equations = {}  # Stores regression models for each filter
        self.filter_importances = {}    # Stores filter importance scores
        logger.info('StructuralEquations initialized.')
        # Initialize filter importances for all layers
        for idx in self.cnn_scm.dag.keys():
            num_filters = len(self.cnn_scm.dag[idx]['filters'])
            self.filter_importances[idx] = np.zeros(num_filters)

    def collect_data(self, data_loader):
        logger.info('Starting data collection for structural equations...')
        start_time = time.time()
        # Collect transformed outputs for all filters
        self.filter_outputs = {}
        self.parent_outputs = {}

        # Hook function to collect outputs
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Register hooks
        activation = {}
        hooks = []
        for idx, (name, layer) in enumerate(self.cnn_scm.conv_layers):
            hooks.append(layer.register_forward_hook(get_activation(name)))

        # Pass data through the model
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                _ = self.cnn_scm.model(images)
                # Collect activations
                for idx, (name, layer) in enumerate(self.cnn_scm.conv_layers):
                    output = activation[name]
                    transformed = frobenius_norm(output).cpu().numpy()
                    if idx not in self.filter_outputs:
                        self.filter_outputs[idx] = transformed
                    else:
                        self.filter_outputs[idx] = np.concatenate((self.filter_outputs[idx], transformed), axis=0)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Collect parent outputs
        for idx in self.cnn_scm.dag.keys():
            parents = self.cnn_scm.dag[idx]['parents']
            if parents:
                parent_idx = parents[0]  # Assuming one parent layer
                self.parent_outputs[idx] = self.filter_outputs[parent_idx]
            else:
                self.parent_outputs[idx] = None  # Input layer has no parents

        total_time = time.time() - start_time
        logger.info(f'Data collection completed in {total_time:.2f}s')

    def learn_structural_equations(self):
        logger.info('Starting learning of structural equations...')
        start_time = time.time()
        # For each layer, train regression models and accumulate importance scores for parent filters
        for idx in tqdm(self.cnn_scm.dag.keys()):
            y = self.filter_outputs[idx]
            X = self.parent_outputs[idx]

            if X is not None:
                # Flatten X and y
                X_flat = X.reshape(-1, X.shape[-1])
                y_flat = y.reshape(-1, y.shape[-1])

                num_parent_filters = X_flat.shape[1]
                num_child_filters = y_flat.shape[1]
                importance = np.zeros(num_parent_filters)

                # Train linear regression for each child filter and accumulate importance scores
                for i in range(num_child_filters):
                    model_reg = LinearRegression()
                    model_reg.fit(X_flat, y_flat[:, i])
                    coeffs = np.abs(model_reg.coef_)  # Absolute value of coefficients
                    importance += coeffs  # Accumulate importance scores for parent filters

                # Accumulate importance scores for the parent layer
                parent_idx = self.cnn_scm.dag[idx]['parents'][0]
                self.filter_importances[parent_idx] += importance
            else:
                # No parents; skip accumulation
                pass

        # Normalize importance scores for each layer
        for idx in self.filter_importances.keys():
            importance = self.filter_importances[idx]
            total = np.sum(importance)
            if total != 0:
                self.filter_importances[idx] = importance / total
            else:
                num_filters = len(self.filter_importances[idx])
                self.filter_importances[idx] = np.ones(num_filters) / num_filters

        total_time = time.time() - start_time
        logger.info(f'Learning structural equations completed in {total_time:.2f}s')

    def get_filter_importances(self):
        # Return the filter importances
        return self.filter_importances

# Instantiate StructuralEquations and learn models
struct_eq = StructuralEquations(cnn_scm)

# Collect data for structural equations
struct_eq.collect_data(train_loader)

# Learn structural equations
struct_eq.learn_structural_equations()

# Get filter importances
filter_importances = struct_eq.get_filter_importances()

# Function to prune filters by importance
def prune_filters_by_importance(model, cnn_scm, filter_importances, percent):
    logger.info(f'Starting pruning {percent}% of filters by importance...')
    start_time = time.time()
    pruned_model = copy.deepcopy(model)
    num_filters_to_prune = {}
    filters_to_prune = {}

    # Prune filters based on importance scores in each layer
    for idx in cnn_scm.dag.keys():
        importance = filter_importances.get(idx)
        if importance is not None:
            num_filters = len(importance)
            n_prune = int(num_filters * percent / 100)
            if n_prune == 0 and percent > 0:
                n_prune = 1  # Ensure at least one filter is pruned if percent > 0
            num_filters_to_prune[idx] = n_prune
            # Get indices of filters with lowest importance
            prune_indices = np.argsort(importance)[:n_prune]
            filters_to_prune[idx] = prune_indices
        else:
            num_filters_to_prune[idx] = 0
            filters_to_prune[idx] = []

    # Prune filters by zeroing out their weights
    with torch.no_grad():
        for idx, prune_indices in filters_to_prune.items():
            name = cnn_scm.dag[idx]['name']
            layer = get_submodule(pruned_model, name)
            for filter_idx in prune_indices:
                if filter_idx >= layer.weight.shape[0]:
                    logger.warning(f'Filter index {filter_idx} out of bounds for layer {name}')
                    continue
                layer.weight[filter_idx] = 0
                if layer.bias is not None:
                    layer.bias[filter_idx] = 0

    total_time = time.time() - start_time
    logger.info(f'Pruning by importance completed in {total_time:.2f}s')
    return pruned_model

# Function to prune random filters
def prune_random_filters(model, cnn_scm, percent):
    logger.info(f'Starting pruning {percent}% of filters randomly...')
    start_time = time.time()
    pruned_model = copy.deepcopy(model)
    num_filters_to_prune = {}
    filters_to_prune = {}

    # Calculate the number of filters to prune in each layer
    for idx in cnn_scm.dag.keys():
        num_filters = cnn_scm.dag[idx]['layer'].out_channels
        n_prune = int(num_filters * percent / 100)
        if n_prune == 0 and percent > 0:
            n_prune = 1  # Ensure at least one filter is pruned if percent > 0
        num_filters_to_prune[idx] = n_prune
        # Randomly select filters to prune
        prune_indices = random.sample(range(num_filters), n_prune)
        filters_to_prune[idx] = prune_indices

    # Prune filters by zeroing out their weights
    with torch.no_grad():
        for idx, prune_indices in filters_to_prune.items():
            name = cnn_scm.dag[idx]['name']
            layer = get_submodule(pruned_model, name)
            for filter_idx in prune_indices:
                layer.weight[filter_idx] = 0
                if layer.bias is not None:
                    layer.bias[filter_idx] = 0

    total_time = time.time() - start_time
    logger.info(f'Random pruning completed in {total_time:.2f}s')
    return pruned_model

# Function to evaluate the model accuracy
def evaluate_model(model, test_loader):
    logger.info('Starting model evaluation...')
    start_time = time.time()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    total_time = time.time() - start_time
    logger.info(f'Model evaluation completed in {total_time:.2f}s')
    return accuracy

# Lists to store accuracies
importance_accuracies = []
random_accuracies = []
prune_percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Baseline accuracy before pruning
baseline_accuracy = evaluate_model(model, test_loader)
logger.info(f'Baseline Accuracy: {baseline_accuracy * 100:.2f}%')

# Iterate over different pruning percentages
for percent in prune_percentages:
    # Prune filters by importance
    pruned_model_importance = prune_filters_by_importance(model, cnn_scm, filter_importances, percent)
    # Evaluate pruned model
    acc_importance = evaluate_model(pruned_model_importance, test_loader)
    importance_accuracies.append(acc_importance * 100)
    logger.info(f'Accuracy after pruning {percent}% by importance: {acc_importance * 100:.2f}%')

    # Prune filters randomly
    pruned_model_random = prune_random_filters(model, cnn_scm, percent)
    # Evaluate pruned model
    acc_random = evaluate_model(pruned_model_random, test_loader)
    random_accuracies.append(acc_random * 100)
    logger.info(f'Accuracy after pruning {percent}% randomly: {acc_random * 100:.2f}%')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot([0] + prune_percentages, [baseline_accuracy * 100] + importance_accuracies, marker='o', label='Pruning by Importance')
plt.plot([0] + prune_percentages, [baseline_accuracy * 100] + random_accuracies, marker='s', label='Random Pruning')
plt.title('ResNet18 Accuracy vs. Percentage of Filters Pruned')
plt.xlabel('Percentage of Filters Pruned')
plt.ylabel('Accuracy (%)')
plt.xticks(prune_percentages)
plt.legend()
plt.grid(True)
plt.show()
logger.info('Experiment completed.')

# Additional Plots
logger.info('Plotting importance distribution and top/bottom filters.')

# Collect all importance scores
all_importances = []
for idx in filter_importances.keys():
    importance = filter_importances[idx]
    all_importances.extend(importance)

# Plot importance distribution across all the network
plt.figure(figsize=(10, 6))
plt.hist(all_importances, bins=50, edgecolor='black')
plt.title('Importance Distribution Across All Filters')
plt.xlabel('Importance Score')
plt.ylabel('Number of Filters')
plt.grid(True)
plt.show()

# Get the top 5 filters by importance
flattened_importances = []
filter_identifiers = []
for idx in filter_importances.keys():
    importance = filter_importances[idx]
    for i, imp in enumerate(importance):
        flattened_importances.append(imp)
        filter_identifiers.append((idx, i))

# Sort filters by importance
sorted_indices = np.argsort(flattened_importances)[::-1]  # Descending order
top_5_indices = sorted_indices[:5]
top_5_importances = [flattened_importances[i] for i in top_5_indices]
top_5_filters = [filter_identifiers[i] for i in top_5_indices]

# Plot top 5 filters by importance
plt.figure(figsize=(16, 12))
plt.bar(range(5), top_5_importances, tick_label=[f'Layer {idx}, Filter {i}' for idx, i in top_5_filters])
plt.title('Top 5 Filters by Importance')
plt.xlabel('Filter')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Get the top 5 filters with the lowest importance
bottom_5_indices = sorted_indices[-5:]
bottom_5_importances = [flattened_importances[i] for i in bottom_5_indices]
bottom_5_filters = [filter_identifiers[i] for i in bottom_5_indices]

# Plot top 5 filters with the lowest importance
plt.figure(figsize=(16, 12))
plt.bar(range(5), bottom_5_importances, tick_label=[f'Layer {idx}, Filter {i}' for idx, i in bottom_5_filters])
plt.title('Top 5 Filters with Lowest Importance')
plt.xlabel('Filter')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
