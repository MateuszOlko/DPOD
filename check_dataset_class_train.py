from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision.transforms import ToPILImage


def main(kaggle_dataset_dir_path, show=False, save=False):

    dataset = KaggleImageMaskDataset(kaggle_dataset_dir_path, is_train=True)
    models_handler = ModelsHandler(kaggle_dataset_dir_path)

    # useful if you dont have all the masks
    some_mask_paths = glob(os.path.join(
        kaggle_dataset_dir_path, 'train_targets/*.npy'
    ))[:3]
    some_image_ids = [os.path.split(x)[1][:-4] for x in some_mask_paths]
    if not some_image_ids:
        raise KeyError('no masks found')

    if save:
        output_dir = f"sanity_checks/{time.strftime('%Y.%m.%dT%H:%M')}"
        os.makedirs(output_dir, exist_ok=True)

    for image_id in some_image_ids:
        id_ = list(dataset.images_ID).index(image_id)
        print(id_)
        # load
        image, masks, prediction_string = dataset[id_]

        # to numpy format

        mask = np.stack(masks, axis=-1).astype(np.uint8)

        print('unique classes', np.unique(mask[..., 0]))

        mask[masks[0] >= dataset.num_of_models, 0] = -1

        overlay_img, model_type_img, height_img, angle_img = \
            models_handler.make_visualizations(image, mask)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].imshow(image)
        axs[0, 1].imshow(overlay_img)
        axs[1, 0].imshow(model_type_img)
        axs[1, 1].imshow(angle_img)
        plt.tight_layout()

        if save:
            plt.savefig(f'{output_dir}/loaded_masks-{id_}.jpg')

        if show:
            plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser(description="run to check whether dataset return what it should")
    arg_parser.add_argument('kaggle_dataset_dir_path')
    arg_parser.add_argument('--show', action='store_true', help='show masks visualization with matplotlib (blocks)')
    arg_parser.add_argument('--save', action='store_true', help='save produced visualizations in sanity_checks')

    args = arg_parser.parse_args()
    main(args.kaggle_dataset_dir_path, args.show, args.save)