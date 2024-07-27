import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


class TestFilterImportanceSCM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class with the model, data, and the SCM tool."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        model = SmallCNN().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def train_model(model, train_loader, criterion, optimizer, epochs=10):
            """Train the model on the training data."""
            model.train()
            for epoch in range(epochs):
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

        train_model(model, trainloader, criterion, optimizer)

        cls.model = model
        cls.trainloader = trainloader
        cls.testloader = testloader
        cls.device = device
        cls.scm_tool = FilterImportanceSCM(model, trainloader, testloader)
        cls.scm_tool.run_scm()

    def test_measure_accuracy(self):
        """Test the accuracy measurement of the model."""
        accuracy = self.scm_tool.measure_accuracy()
        self.assertGreater(accuracy, 0.1)  # Check if accuracy is greater than 10%

    def test_zero_least_important_filters(self):
        """Test zeroing the least important filters."""
        initial_accuracy = self.scm_tool.measure_accuracy()
        self.scm_tool.zero_least_important_filters(self.model.conv2, percentage=1)
        post_prune_accuracy = self.scm_tool.measure_accuracy()
        self.assertLess(
            post_prune_accuracy, initial_accuracy
        )  # Check if accuracy decreases

    def test_zero_random_filters(self):
        """Test zeroing random filters."""
        initial_accuracy = self.scm_tool.measure_accuracy()
        self.scm_tool.zero_random_filters(self.model.conv2, percentage=1)
        post_prune_accuracy = self.scm_tool.measure_accuracy()
        self.assertLess(
            post_prune_accuracy, initial_accuracy
        )  # Check if accuracy decreases


if __name__ == "__main__":
    unittest.main()
