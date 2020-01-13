import os
import torch
import datetime

EXPERIMENTS_DIR = "../experiments"


def get_mask(tensor):
    """
    Args:
        tensor: prediction tensor of dim (batch_size, classes, H, W)

    Returns: mask of dim (batch_size, H, W)
    """
    return torch.argmax(tensor, dim=1)


def get_experiment_directory(args):
    now = f"{datetime.datetime.now():%b-%d-%H:%M}" 
    exp_path = os.path.join(EXPERIMENTS_DIR, args.name, now)
    os.makedirs(exp_path, exist_ok=True)
    return exp_path
