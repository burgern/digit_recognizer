import numpy as np
import torch
from typing import Optional
from pathlib import Path
import dataclasses
import wandb


__repo_path__ = Path(__file__).parent.parent
__data_path__ = __repo_path__ / 'data'
__logs_path__ = __repo_path__ / 'wandb'


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


def setup_wandb(cfg, cfg_class):
    # set up logging directory
    out_path = __repo_path__ / 'out' / cfg.wandb_project
    out_path.mkdir(exist_ok=True, parents=True)

    # initialize wandb
    run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, dir=out_path)

    # update config with given wandb parameters (if any, e.g. sweep)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.update(wandb.config)
    cfg = cfg_class(**cfg_dict)

    # try all update methods
    wandb.config.update(cfg_dict)
    run.config.update(cfg_dict)

    return run, cfg
