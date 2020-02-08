import numpy as np
from torch.utils.data import Subset

from .kaggle_dataset import KaggleImageMaskDataset
from .linemod_dataset import LinemodImageMaskDataset


PATHS = {
    "kaggle": "/mnt/bigdisk/datasets/kaggle",
    "linemod": "/mnt/bigdisk/datasets/linemod",
}


def make_dataset(args, name):
    if "kaggle" in name:
        kaggle_dataset = KaggleImageMaskDataset(PATHS['kaggle'], setup="all")
        train_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="train")
        val_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="val")
        return train_data, val_data, kaggle_dataset
    if "linemod" in name:
        whole_dataset = LinemodImageMaskDataset(PATHS['linemod'], setup="all")
        train_data = LinemodImageMaskDataset(PATHS['linemod'], setup="train")
        val_data = LinemodImageMaskDataset(PATHS['linemod'], setup="val")
        return train_data, val_data, whole_dataset
    else:
        raise AttributeError(f"Dataset \"{name}\" is not supported!")
        
def make_test_dataset():
    return KaggleImageMaskDataset(PATHS['kaggle'], is_train=False)
