import numpy as np
import torch
from typing import Optional


def set_seed(seed: Optional[int] = None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_device(gpu: bool):
    if gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise ValueError
    else:
        device = 'cpu'
    print(f'Device: {device}')
    return device
