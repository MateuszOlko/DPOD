import numpy as np
from torch.utils.data import Subset

from .kaggle_dataset import KaggleImageMaskDataset


PATHS = {
    "kaggle": "/mnt/bigdisk/datasets/kaggle"
}


def make_dataset(args, name):
    if "kaggle" in name:
        kaggle_dataset = KaggleImageMaskDataset(PATHS['kaggle'])

        indices = np.arange(len(kaggle_dataset))
        np.random.shuffle(indices)

        train_data = Subset(kaggle_dataset, indices[int(len(indices) * args.val_size):])
        val_data = Subset(kaggle_dataset, indices[:int(len(indices) * args.val_size)])

        return train_data, val_data, kaggle_dataset
    else:
        raise AttributeError(f"Dataset \"{name}\" is not supported!")
        
def make_test_dataset():
    return KaggleImageMaskDataset(PATHS['kaggle'], is_train=False)
