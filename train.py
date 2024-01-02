from torch.utils.data import DataLoader
from src import get_cfg, load_train_data, set_seed, split_set, DigitRecognizerDataset


def main():
    # load configuration (including command line arguments)
    cfg = get_cfg()

    # set seed for reproducibility
    set_seed(cfg.seed)

    # load data
    data, labels = load_train_data()

    # split data into train and validation
    train_data, train_labels, val_data, val_labels = split_set(data, labels, cfg.train_validation_split, cfg.seed)

    # create train and validation datasets
    train_dataset = DigitRecognizerDataset(train_data, train_labels)
    val_dataset = DigitRecognizerDataset(val_data, val_labels)

    # create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

    # create model and optimizer

    # initialize weights and biases

    print('hello')


if __name__ == '__main__':
    main()
