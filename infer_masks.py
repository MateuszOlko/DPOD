from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.datasets.linemod import LinemodImageMaskDataset
from DPOD.datasets import PATHS
import os
import numpy as np
from argparse import ArgumentParser


def main(path_to_model, path_to_output_dir, setup="test", debug=False):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare dataset
    dataset = LinemodImageMaskDataset(PATHS['linemod'], setup=setup)

    # load correspondence block
    model = DPOD(image_size=(480, 640), num_classes=7+1)
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval()

    infer_masks(model, dataset, path_to_output_dir, debug, device)

def infer_masks(model, dataset, path_to_output_dir, debug=False, device="cpu"):
    # prepare output dir
    os.makedirs(path_to_output_dir, exist_ok=True)

    # prepare data
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8
    )

    # debug mode
    n_images_to_process = 20 if debug else 100000

    skipped = 0
    with torch.no_grad():
        for n_image in trange(len(dataset)):
            if n_image >= n_images_to_process:
                break
            
            image_id = dataset.images_filenames[n_image]
            path = os.path.join(
                path_to_output_dir,
                image_id[:-4][6:]+"_masks.npy"
            )

            if os.path.exists(path) and not debug:
                skipped += 1
                continue
            
            images, _ = dataset[n_image] 
            images = images.to(device)
            images = images.view((1, *images.shape))

            class_mask, u_channel, v_channel = model(images)

            # get rid of batch dimension
            class_mask = class_mask[0]
            u_channel  = u_channel[0]
            v_channel  = v_channel[0]

            # select most probable class/colour
            class_mask = torch.argmax(class_mask, dim=0)
            u_channel  = torch.argmax(u_channel,  dim=0)
            v_channel  = torch.argmax(v_channel,  dim=0)

            # reformat to single numpy array and save
            array = torch.stack([class_mask, u_channel, v_channel]).cpu().numpy().astype(np.uint8)
            np.save(path, array)
    
    print(f"Skipped processing of {skipped} images - already processed")


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=
    """infers mask using provided model and saves them as path_to_output_dir/ImageId.npy
    as (3,h,w) np.uint8 numpy arrays, ignores already present masks""")
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_output_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')
    arg_parser.add_argument('-s', '--setup', default='test', help='With witch setup initialize dataset')
    args = arg_parser.parse_args()

    main(args.path_to_model, args.path_to_output_dir, args.setup, args.debug)
