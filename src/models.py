import torch
import torch.nn as nn
import torch.nn.functional as F


def train(model, optimizer, criterion, train_dataloader, device):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    train_correct = 0

    for data, labels in train_dataloader:
        data, labels = data.to(device), labels.to(device)  # Move data to the appropriate device (GPU or CPU)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(data)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model

        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataloader.dataset)
    train_accuracy = 100. * train_correct / len(train_dataloader.dataset)

    return train_loss, train_accuracy


def validate(model, criterion, val_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, labels in val_dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataloader.dataset)
    val_accuracy = 100. * val_correct / len(val_dataloader.dataset)

    return val_loss, val_accuracy


def initialize_weights_he(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)


class CNN(nn.Module):
    def __init__(self, init_weights: bool = True):
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel (grayscale image), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: 64 * 7 * 7 input features (assuming input images are 28x28), 128 output features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Output layer: 128 input features, 10 output features for 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

        # Initialize weights
        if init_weights:
            self.apply(initialize_weights_he)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)

        # Apply first fully connected layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply output layer
        x = self.fc2(x)

        return x
