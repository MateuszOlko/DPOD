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
    min_inliers = args.min_inliers
    no_class = args.no_class
    solvePnPRansacKwargs = dict()
    for key in ['iterationsCount', 'reprojectionError', 'confidence', 'flags']:
        val = getattr(args, key)
        if val:
            solvePnPRansacKwargs[key] = val

    ransac_block = PoseBlock(PATHS['kaggle'], min_inliers=min_inliers, no_class=no_class, **solvePnPRansacKwargs)

    print("Infering masks")
    infer_masks(model, val_data, args.path_to_masks_dir, args.debug, device)

    masks_paths = glob(f'{args.path_to_masks_dir}/*.npy')  # locate masks to process further

    print("Applying ransac")
    apply_ransac(masks_paths, ransac_block, args.path_to_outputs_dir, args.debug)

    print("Making submission")
    path_to_submission_file = os.path.join(
        args.path_to_outputs_dir,
        'submission.csv'
    )
    make_submission_from_ransac_directory(
        args.path_to_outputs_dir,
        path_to_submission_file
    )
    loss = calculate_loss_of_prediction(path_to_submission_file, os.path.join(PATHS['kaggle'], "train.csv"))
    print("Loss on train date set:", loss)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="""
    performs infer_masks to path_masks_dir, 
    and masks2instances and instances2submission to path_outputs_dir
    generating submission named submission.csv
    on path_outputs_dir using model saved under path_to_model
    """)
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_masks_dir')
    arg_parser.add_argument('path_outputs_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')

    # our ransac parameters
    arg_parser.add_argument('--min_inliers', type=int, default=50)
    arg_parser.add_argument('--no_class', action='store_true')

    # cv2.solvePnPRansac kwargs
    arg_parser.add_argument('--iterationsCount', type=int)
    arg_parser.add_argument('--reprojectionError', type=float)
    arg_parser.add_argument('--confidence', type=float)
    arg_parser.add_argument('--flags', type=str)

    args = arg_parser.parse_args()

    main(args)
