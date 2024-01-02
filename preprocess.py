import pandas as pd
import numpy as np
import torch
import cv2
import argparse

from src import __data_path__


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess the train.csv / test.csv provided by Kaggle.')
    parser.add_argument('train_path', type=str, help='Path to the train.csv file.')
    parser.add_argument('test_path', type=str, help='Path to the test.csv file.')
    args = parser.parse_args()

    # Load train.csv and test.csv using pandas
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # Extract labels as NumPy arrays
    train_labels = train_df.iloc[:, 0].values.astype(np.int64)

    # Extract pixel values as NumPy arrays
    train_data = train_df.iloc[:, 1:].values.astype(np.float32)
    test_data = test_df.iloc[:, :].values.astype(np.float32)

    # Normalize pixel values to the range [0, 1]
    train_data /= 255.0
    test_data /= 255.0

    # Convert NumPy arrays to PyTorch tensors
    train_data_tensor = torch.from_numpy(train_data).reshape((-1, 1, 28, 28))
    test_data_tensor = torch.from_numpy(test_data).reshape((-1, 1, 28, 28))
    train_labels_tensor = torch.from_numpy(train_labels)

    # create data, test and train directories with path.mkdir(exist_ok=True)
    __data_path__.mkdir(exist_ok=True)
    (__data_path__ / 'train').mkdir(exist_ok=True)
    (__data_path__ / 'test').mkdir(exist_ok=True)
    (__data_path__ / 'train/imgs').mkdir(exist_ok=True)
    (__data_path__ / 'test/imgs').mkdir(exist_ok=True)

    # # Save the tensors for future use
    torch.save(train_data_tensor, __data_path__ / 'train/train_data.pt')
    torch.save(test_data_tensor, __data_path__ / 'test/test_data.pt')
    torch.save(train_labels_tensor, __data_path__ / 'train/train_labels.pt')

    # Save images as PNGs
    for data_set, data in [('train', train_data), ('test', test_data)]:
        for i, img in enumerate(data):
            img = img.reshape((28, 28))
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(__data_path__ / f'{data_set}/imgs/{str(i).zfill(5)}.png'), img)

    # Save train labels as CSV
    train_labels_df = pd.DataFrame({'Label': train_labels})
    train_labels_df.index = np.arange(len(train_labels_df))
    train_labels_df.index.name = 'ImageId'
    train_labels_df.to_csv(__data_path__ / 'train/train_labels.csv', index=True, header=True)


if __name__ == '__main__':
    main()
