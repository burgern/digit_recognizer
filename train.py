import torch
from torch.optim import Adam
import torch.nn as nn
import wandb
from pathlib import Path
import dataclasses
from tqdm import tqdm
from src import (get_cfg, load_train_data, set_seed, CNN, get_device, get_train_val_dataloaders, train, validate)


def main():
    # Load configuration (including command line arguments)
    cfg = get_cfg()
    device = get_device(cfg.gpu)
    set_seed(cfg.seed)

    # Load data
    data, labels = load_train_data()
    train_dataloader, val_dataloader = get_train_val_dataloaders(data, labels, cfg.train_validation_split,
                                                                 cfg.batch_size, cfg.seed)

    # Create model, optimizer and criterion
    model = CNN(cfg.init_weights).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Set up logger
    if cfg.debugging:
        wandb.init(project='digit_recognizer', config=dataclasses.asdict(cfg))
    else:
        wandb.init(project='digit_recognizer_debugging', config=dataclasses.asdict(cfg))

    # Start training loop
    for epoch in tqdm(range(cfg.num_epochs), desc='Epochs'):
        # Train and validate
        train_loss, train_accuracy = train(model, optimizer, criterion, train_dataloader, device)
        val_loss, val_accuracy = validate(model, criterion, val_dataloader, device)

        # Log results
        tqdm.write(f'Epoch {epoch + 1}/{cfg.num_epochs}')
        tqdm.write(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'val_loss': val_loss, 'val_accuracy': val_accuracy})

    # Save model
    torch.save(model.state_dict(), str(Path(wandb.run.dir) / 'model.pt'))

    print('hello')


if __name__ == '__main__':
    main()
