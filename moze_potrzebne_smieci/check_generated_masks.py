from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from DPOD.models_handler import ModelsHandler


def main(kaggle_dataset_dir_path, show=False, save=False):
    """
        Loads some masks from kaggle_dataset_dir_path
        and draws overlays with ModelsHandler

        This verifies whether mask can by interpreted by ModelsHandler
        and ModelsHandler behaviour
    """

    models_handler = ModelsHandler(kaggle_dataset_dir_path)

    some_mask_paths = glob(os.path.join(
        kaggle_dataset_dir_path, 'train_targets/*.npy'
    ))[:3]

    if save:
        output_dir = f"sanity_checks/{time.strftime('%Y.%m.%dT%H:%M')}"
        os.makedirs(output_dir, exist_ok=True)

    for n, mask_path in enumerate(some_mask_paths):
        mask = np.load(mask_path)
        image_id = os.path.split(mask_path[:-4])[-1]
        image = cv2.imread(
            os.path.join(
                kaggle_dataset_dir_path, 'train_images', image_id+'.jpg'
            ),
        )[:, :, ::-1]

        print(image_id)

        overlay_img, model_type_img, height_img, angle_img = \
            models_handler.make_visualizations(image, mask)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].imshow(overlay_img)
        axs[0, 1].imshow(model_type_img)
        axs[1, 0].imshow(height_img)
        axs[1, 1].imshow(angle_img)
        plt.tight_layout()

        if save:
            plt.savefig(f'{output_dir}/generated_masks-{n}.jpg')

        if show:
            plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser(description="run to check whether mask are read and interpreted properly")
    arg_parser.add_argument('kaggle_dataset_dir_path')
    arg_parser.add_argument('--show', action='store_true', help='show masks visualization with matplotlib (blocks)')
    arg_parser.add_argument('--save', action='store_true', help='save produced visualizations in sanity_checks')

    args = arg_parser.parse_args()
    main(args.kaggle_dataset_dir_path, args.show, args.save)
