from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from src import (get_cfg, load_train_data, set_seed, split_set, DigitRecognizerDataset, CNN, get_device,
                 initialize_weights_he)


def main():
    # Load configuration (including command line arguments)
    cfg = get_cfg()

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Load device (gpu or cpu)
    device = get_device(cfg.gpu)

    # Load data
    data, labels = load_train_data()

    # Split data into train and validation
    train_data, train_labels, val_data, val_labels = split_set(data, labels, cfg.train_validation_split, cfg.seed)

    # Create train and validation datasets
    train_dataset = DigitRecognizerDataset(train_data, train_labels)
    val_dataset = DigitRecognizerDataset(val_data, val_labels)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    # Create model, optimizer and criterion
    model = CNN().to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize model weights
    if cfg.init_weights:
        model.apply(initialize_weights_he)

    # Initialize logging using wandb

    # Start training loop
    for epoch in range(cfg.num_epochs):
        # Training phase
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

        # Validation phase
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

        print(f'Epoch {epoch + 1}/{cfg.num_epochs}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    print('hello')


if __name__ == '__main__':
    main()
