from dataclasses import dataclass
from simple_parsing import ArgumentParser
import torch
import pandas as pd
from src import CNN, load_test_data


@dataclass
class Config:
    model_path: str  # path to model to load
    seed: int = 42  # random seed for reproducibility
    gpu: bool = True  # whether to use GPU or not


def main():
    # Load configuration (incl. cli)
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")
    cfg = parser.parse_args().config

    # Load data
    data = load_test_data()

    # Load model from path
    model = CNN()
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval()

    # Infer predictions on data
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

    # Save predictions to pandas dataframe
    df = pd.DataFrame({'ImageId': range(1, len(predicted) + 1), 'Label': predicted})
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
