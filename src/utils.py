import numpy as np
import torch
from typing import Optional


def set_seed(seed: Optional[int] = None):
    if seed is None:
        seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
