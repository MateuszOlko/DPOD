from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.datasets import PATHS
import os
import numpy as np
from argparse import ArgumentParser


def main(path_to_model, path_to_output_dir, debug=False):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = KaggleImageMaskDataset(PATHS['kaggle'], is_train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1
    )

    # load correspondence block
    model = DPOD(image_size=(2710 // 8, 3384 // 8))
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)

    # prepare output dir
    os.makedirs(path_to_output_dir, exist_ok=True)

    # debug mode
    n_images_to_process = 20 if debug else 100000

    with torch.no_grad():
        for n_image, images in enumerate(tqdm(data_loader)):
            if n_image >= n_images_to_process:
                break

            image_id = dataset.get_IDs()[n_image]
            path = os.path.join(
                path_to_output_dir,
                f'{image_id}.npy'
            )

            if os.path.exists(path):
                continue

            images = images.to(device)

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


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=
    """infers mask using provided model and saves them as path_to_output_dir/ImageId.npy
    as (3,h,w) np.uint8 numpy arrays, ignores already present masks""")
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_output_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')
    args = arg_parser.parse_args()

    main(args.path_to_model, args.path_to_output_dir, args.debug)