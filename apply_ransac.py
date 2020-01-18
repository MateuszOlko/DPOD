from tqdm import tqdm
import torch
from DPOD.model import PoseBlock
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
import pickle
from DPOD.datasets import PATHS
import json


def main(args):
    solvePnPRansacKwargs = dict()
    for key in ['iterationsCount', 'reprojectionError', 'confidence', 'flags']:
        val = getattr(args, key)
        if val:
            solvePnPRansacKwargs[key] = val

    masks_paths = glob(f'{args.path_to_masks_dir}/*.npy')

    # load ransac block
    ransac_block = PoseBlock(PATHS['kaggle'], min_inliers=args.min_inliers, no_class=args.no_class, **solvePnPRansacKwargs)

    apply_ransac(masks_paths, ransac_block, args.path_to_output_dir, args.debug)

def apply_ransac(masks_paths, ransac_block, path_to_output_dir, debug=False):
    # prepare output dir
    os.makedirs(path_to_output_dir, exist_ok=True)

    n_masks_to_process = 20 if debug else 100000

    skipped = 0
    with torch.no_grad():
        for n_mask, mask_path in enumerate(tqdm(masks_paths)):
            if n_mask >= n_masks_to_process:
                break

            image_id = os.path.split(mask_path)[1][:-4]
            output_file_path = os.path.join(
                path_to_output_dir,
                f'{image_id}_instances.pkl'
            )

            if os.path.exists(output_file_path) and not debug:
                skipped += 1
                continue

            tensor = torch.tensor(np.load(mask_path))
            class_, u_channel, v_channel = tensor
            ransac_instances = ransac_block(class_, u_channel, v_channel)

            with open(output_file_path, 'wb') as file:
                pickle.dump(ransac_instances, file)
    print(f"Skipped processing of {skipped} images - already processed")

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="""
    Applies ransac with specified parameters to all (3,h,w) np.uint8 
    path_to_masks_dir/<ImageId>.npy masks with class, u, v channels
    and saves (output of PoseBlock)
    [
        [
            model_id,                     int
            ransac_translation_vector,    (3)   float np.array
            ransac_rotation_matrix,       (3,3) float np.array
        ]
        for each instance found
    ]
    to path_to_output_dir/<ImageId>_instances.pkl with pickle.dump
    skipping calculating already saved outputs
    """)
    arg_parser.add_argument('path_to_masks_dir')
    arg_parser.add_argument('path_to_output_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')

    arg_parser.add_argument('--min_inliers', type=int, default=50)
    arg_parser.add_argument('--no_class', action='store_true')

    arg_parser.add_argument('--iterationsCount', type=int)
    arg_parser.add_argument('--reprojectionError', type=float)
    arg_parser.add_argument('--confidence', type=float)
    arg_parser.add_argument('--flags', type=str)

    args = arg_parser.parse_args()
    main(args)


