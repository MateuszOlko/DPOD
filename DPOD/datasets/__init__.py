import numpy as np
from torch.utils.data import Subset

from .kaggle_dataset import KaggleImageMaskDataset


PATHS = {
    "kaggle": "/mnt/bigdisk/datasets/kaggle"
}


def make_dataset(args, name):
    if "kaggle" in name:
        kaggle_dataset = KaggleImageMaskDataset(PATHS['kaggle'], setup="all")
        train_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="train")
        val_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="val")
        return train_data, val_data, kaggle_dataset
    else:
        raise AttributeError(f"Dataset \"{name}\" is not supported!")
        
def make_test_dataset():
    return KaggleImageMaskDataset(PATHS['kaggle'], is_train=False)
