import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from applybn.explainable.scm_nn_importance import FilterImportanceSCM


class SmallCNN(nn.Module):
    def __init__(self):
        """Initialize the SmallCNN model with three convolutional layers and two fully connected layers."""
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Define the forward pass of the model."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Main experiment
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = datasets.CIFAR10(
    root="../../data", train=True, download=True, transform=transform
)
testset = datasets.CIFAR10(
    root="../../data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

model = SmallCNN()

# Training the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    """
    Train the model on the training data.

    Parameters:
    model (nn.Module): The neural network model.
    train_loader (DataLoader): DataLoader for the training data.
    criterion (nn.Module): The loss function.
    optimizer (optim.Optimizer): The optimizer.
    epochs (int): The number of epochs to train the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0


train_model(model, trainloader, criterion, optimizer)

scm_tool = FilterImportanceSCM(model, trainloader, testloader)
scm_tool.run_scm()

# Measure accuracy before pruning
accuracy_before = scm_tool.measure_accuracy()
print(f"Accuracy before pruning: {accuracy_before}")

# Iteratively prune 1% of the least important filters and measure accuracy
pruning_iterations = 20
accuracies_importance = [accuracy_before]
accuracies_random = [accuracy_before]
percent_pruned = [0]

# Clone the model for random pruning
model_random = SmallCNN().to(scm_tool.device)
model_random.load_state_dict(model.state_dict())

scm_tool_random = FilterImportanceSCM(model_random, trainloader, testloader)
scm_tool_random.run_scm()

for i in range(pruning_iterations):
    # Importance-based pruning
    scm_tool.zero_least_important_filters(model.conv2, percentage=i)
    accuracy_importance = scm_tool.measure_accuracy()
    accuracies_importance.append(accuracy_importance)

    # Random pruning
    scm_tool_random.zero_random_filters(model_random.conv2, percentage=i)
    accuracy_random = scm_tool_random.measure_accuracy()
    accuracies_random.append(accuracy_random)

    percent_pruned.append((i + 1) * 1)
    print(f"Iteration {i + 1}, Percent Pruned: {(i + 1) * 1}%")
    print(f"  Accuracy (Importance): {accuracy_importance}")
    print(f"  Accuracy (Random): {accuracy_random}")

# Plot accuracy vs percent of filters pruned
plt.plot(
    percent_pruned, accuracies_importance, marker="o", label="Importance-based Pruning"
)
plt.plot(percent_pruned, accuracies_random, marker="x", label="Random Pruning")
plt.xlabel("Percent of Filters Pruned")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Percent of Filters Pruned")
plt.legend()
plt.grid(True)
plt.show()
