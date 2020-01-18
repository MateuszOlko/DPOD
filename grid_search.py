from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.datasets import PATHS
import os
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from glob import glob

from infer_masks import infer_masks
from apply_ransac import apply_ransac
from DPOD.datasets import make_dataset, PATHS
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from instances2submission import make_submission_from_ransac_directory
from mAP import calculate_loss_of_prediction

class JustImageDataset(Dataset):

    def __init__(self, kaggle_validation):
        super().__init__()
        self.kaggle_validation = kaggle_validation
    
    def __len__(self):
        return len(self.kaggle_validation)

    def __getitem__(self, index):
        image, _, _ = self.kaggle_validation[index]
        return image
    
    def get_IDs(self):
        return self.kaggle_validation.get_IDs()

def main(args):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    val_data = KaggleImageMaskDataset(PATHS['kaggle'], setup="val")
    val_data = JustImageDataset(val_data)

    # load correspondence block
    model = DPOD(image_size=(2710 // 8, 3384 // 8))
    model.load_state_dict(torch.load(args.path_to_model))
    model.to(device)

    # Temporary setting, in future will be grid search
    min_inliers = 100
    no_class = True
    solvePnPRansacKwargs = dict()

    masks_paths = glob(f'{args.path_to_masks_dir}/*.npy')
    ransac_block = PoseBlock(PATHS['kaggle'], min_inliers=min_inliers, no_class=no_class, **solvePnPRansacKwargs)

    print("Infering masks")
    infer_masks(model, val_data, args.path_to_masks_dir, args.debug, device)
    print("Applying ransac")
    apply_ransac(masks_paths, ransac_block, args.path_to_ransac_dir, args.debug)
    make_submission_from_ransac_directory(args.path_to_ransac_dir, args.path_to_submission_file)
    loss = calculate_loss_of_prediction(args.path_to_submission_file, os.path.join(PATHS['kaggle'], "train.csv"))
    print(loss)

if __name__ == "__main__":
    arg_parser = ArgumentParser(description=
    """infers mask using provided model and saves them as path_to_ransac_dir/ImageId.npy
    as (3,h,w) np.uint8 numpy arrays, ignores already present masks""")
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_masks_dir')
    arg_parser.add_argument('path_to_ransac_dir')
    arg_parser.add_argument('path_to_submission_file')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')

    args = arg_parser.parse_args()

    main(args)
