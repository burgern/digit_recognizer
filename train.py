import torch
from torch.optim import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, ASGD, LBFGS, Rprop
import torch.nn as nn
import wandb
from pathlib import Path
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from tqdm import tqdm
from src import load_train_data, set_seed, CNN, get_device, get_train_val_dataloaders, train, validate, setup_wandb


optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adagrad': Adagrad
}


@dataclass
class Config:
    # basic options
    seed: int = 42  # random seed for reproducibility
    gpu: bool = True  # whether to use GPU or not

    # wandb options
    wandb: bool = False  # whether to use wandb or not
    wandb_project: str = 'digit_recognizer'  # wandb project name
    wandb_entity: str = 'burgern'  # wandb entity name
    wandb_sweep: bool = False  # whether to use wandb sweep or not - OVERWRITES some of the latter options

    # data options
    train_validation_split: float = 0.8  # train/validation split

    # neural network options
    epochs: int = 3  # number of epochs to train for
    batch_size: int = 64  # batch size for training
    learning_rate: float = 1e-3  # learning rate for optimizer
    optimizer: str = 'adam'  # optimizer to use
    # momentum: float = 0.9  # momentum for optimizer
    # weight_decay: float = 0.0  # weight decay for optimizer
    init_weights: bool = True  # initialize weights using He initialization
    # num_hidden: int = 128  # number of hidden units in the hidden layer
    # dropout: float = 0.2  # dropout probability for dropout layer
    # activation: str = 'relu'  # activation function for hidden layer
    # scheduler: str = 'plateau'  # learning rate scheduler to use
    # patience: int = 10  # patience for learning rate scheduler


def main():
    # Load configuration (incl. cli)
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    cfg = parser.parse_args().config
    device = get_device(cfg.gpu)
    set_seed(cfg.seed)

    # Load data
    data, labels = load_train_data()
    train_dataloader, val_dataloader = get_train_val_dataloaders(data, labels, cfg.train_validation_split,
                                                                 cfg.batch_size, cfg.seed)

    # Create model, optimizer, criterion and scheduler
    model = CNN(cfg.init_weights).to(device)
    optimizer = optimizers[cfg.optimizer](model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                                        patience=cfg.patience, verbose=True)
    # Initialize logger with wandb
    if cfg.wandb:
        wandb_run, cfg = setup_wandb(cfg, Config)

    # Start training loop
    pbar = tqdm(range(cfg.epochs), desc='Epochs')
    for _ in pbar:
        # Train and validate
        train_loss, train_accuracy = train(model, optimizer, criterion, train_dataloader, device)
        val_loss, val_accuracy = validate(model, criterion, val_dataloader, device)

        # Print results
        pbar.set_postfix_str(f'Train Loss / Accuracy: {train_loss:.4f} - {train_accuracy:.2f}% | '
                             f'Val Loss / Accuracy: {val_loss:.4f} - {val_accuracy:.2f}%')
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'val_loss': val_loss, 'val_accuracy': val_accuracy})

    # Save model
    torch.save(model.state_dict(), str(Path(wandb.run.dir) / 'model.pt'))


if __name__ == '__main__':
    main()
