from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler, pnp_ransac_multiple_instances
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
        image, (classification_mask, height_mask, angle_mask), prediction_string = dataset[id_]

        image = image.numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        image = (256 * image).astype(np.uint8)

        instances = pnp_ransac_multiple_instances(height_mask, angle_mask, classification_mask, models_handler, 79, downscaling=8, min_inliers=100)
        print('może')
        for model_id, translation_vector, rotation_matrix in instances:
            print(model_id, translation_vector, rotation_matrix, sep='\n')
            overlay = models_handler.draw_model(np.zeros_like(image), model_id, translation_vector, rotation_matrix)
            plt.imshow(overlay)
            plt.show()
            time.sleep(2)
        plt.imshow(image)
        '''if save:
            plt.savefig(f'{output_dir}/loaded_masks-{id_}.jpg')'''

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