import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from applybn.explainable.scm_nn_importance import NeuronImportanceSCM


class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """Initialize the TabularNN model with fully connected layers."""
        super(TabularNN, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Define the forward pass of the model."""
        return self.network(x)


# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a simple fully connected neural network for tabular data
model = TabularNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], output_dim=2)

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")


train_model(model, trainloader, criterion, optimizer)

# Apply the NeuronImportanceSCM tool
scm_tool = NeuronImportanceSCM(model, trainloader, testloader)
scm_tool.run_scm()

# Clone the model for random pruning
model_random = TabularNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], output_dim=2)
model_random.load_state_dict(model.state_dict())
scm_tool_random = NeuronImportanceSCM(model_random, trainloader, testloader)

# Measure accuracy before pruning
accuracy_before = scm_tool.measure_accuracy()
print(f"Accuracy before pruning: {accuracy_before}")

# Initialize lists to store accuracies
accuracies_importance = [accuracy_before]
accuracies_random = [accuracy_before]
percent_pruned = [0]

# Perform pruning
pruning_iterations = (
    40  # Reduce number of iterations, and prune a fixed percentage each time
)


for i in range(pruning_iterations):
    # Importance-based pruning
    scm_tool.zero_least_important_neurons_across_layers(
        percentage=i
    )  # Fixed 5% pruning per iteration
    accuracy_importance = scm_tool.measure_accuracy()
    accuracies_importance.append(accuracy_importance)

    # Random pruning
    scm_tool_random.zero_random_neurons_across_layers(
        percentage=i
    )  # Fixed 5% pruning per iteration
    accuracy_random = scm_tool_random.measure_accuracy()
    accuracies_random.append(accuracy_random)

    percent_pruned.append((i + 1))  # Adjust the scale for the fixed percentage
    print(f"Iteration {i + 1}, Percent Pruned: {(i + 1)}%")
    print(f"  Accuracy (Importance-based Pruning): {accuracy_importance}")
    print(f"  Accuracy (Random Pruning): {accuracy_random}")


# Plot accuracy vs percent of neurons pruned
plt.plot(
    percent_pruned, accuracies_importance, marker="o", label="Importance-based Pruning"
)
plt.plot(percent_pruned, accuracies_random, marker="x", label="Random Pruning")
plt.xlabel("Percent of Neurons Pruned")
plt.ylabel("Accuracy")
plt.title("Accuracy, NN for tabular data")
plt.legend()
plt.grid(True)
plt.show()
