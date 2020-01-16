import os
import torch
import datetime
import matplotlib.pyplot as plt
import os

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


def tensor1_to_jpg(tensor, path):
    """
    Rescales values to [0, 1] and saves as jpg
    :param tensor: (1, h, w) integer-valued tensor
    """
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    x = tensor.cpu().numpy()
    plt.figure()
    plt.imshow(x)
    plt.savefig(path)


def tensor3_to_jpg(tensor, path):
    """
    Rescales values to [0, 1] and saves as jpg
    :param tensor: (3, h, w) tensor
    """
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    x = tensor.cpu().numpy().transpose(1, 2, 0)
    x = (x - x.min()) / (x.max() - x.min())
    plt.figure()
    plt.imshow(x)
    plt.savefig(path)