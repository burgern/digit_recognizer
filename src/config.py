from dataclasses import dataclass
from simple_parsing import ArgumentParser
from pathlib import Path


__repo_path__ = Path(__file__).parent.parent
__data_path__ = __repo_path__ / 'data'
__logs_path__ = __repo_path__ / 'wandb'


@dataclass
class Config:
    # basic options
    seed: int = 42  # random seed for reproducibility
    gpu: bool = True  # whether to use GPU or not
    debugging: bool = False  # whether to run in debugging mode or not

    # data options
    train_validation_split: float = 0.8  # train/validation split

    # neural network options
    num_epochs: int = 3  # number of epochs to train for
    batch_size: int = 64  # batch size for training
    learning_rate: float = 1e-3  # learning rate for optimizer
    momentum: float = 0.9  # momentum for optimizer
    weight_decay: float = 0.0  # weight decay for optimizer
    init_weights: bool = True  # initialize weights using He initialization
    num_hidden: int = 128  # number of hidden units in the hidden layer
    dropout: float = 0.2  # dropout probability for dropout layer
    activation: str = 'relu'  # activation function for hidden layer
    optimizer: str = 'adam'  # optimizer to use
    loss: str = 'cross_entropy'  # loss function to use
    scheduler: str = 'plateau'  # learning rate scheduler to use


def get_cfg() -> Config:
    # parse config options
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    # return config object
    return args.config
