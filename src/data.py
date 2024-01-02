from torch.utils.data import TensorDataset
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src import __data_path__


def load_train_data() -> (torch.Tensor, torch.Tensor):
    data = torch.load(__data_path__ / 'train/train_data.pt')
    labels = torch.load(__data_path__ / 'train/train_labels.pt')
    return data, labels


def load_test_data() -> torch.Tensor:
    return torch.load(__data_path__ / 'test/test_data.pt')


def split_set(data: torch.Tensor, labels: torch.Tensor, split: float, seed: int = 42) -> (TensorDataset, TensorDataset):
    """ Split the data into sets a and b (split a, (1-split) b) """

    # Create separate indices for each digit (0-9)
    indices_by_digit = [torch.where(labels == i)[0] for i in range(10)]

    # Split each digit's indices into train and validation
    train_indices = []
    val_indices = []
    for indices in indices_by_digit:
        train_indices_digit, val_indices_digit, _, _ = train_test_split(
            indices, indices, test_size=(1 - split), shuffle=True, random_state=seed
        )
        train_indices.extend(train_indices_digit)
        val_indices.extend(val_indices_digit)

    return data[train_indices], labels[train_indices], data[val_indices], labels[val_indices]


class DigitRecognizerDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """ Returns a tuple (data, label), and applies data augmentation """
        data_batch, label_batch = self.data[index], self.labels[index]

        # Data Augmentation
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor()
        ])

        return transform(data_batch), label_batch

    def plot(self, index):
        plt.imshow(self.data[index].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {self.labels[index]}')
        plt.show()
