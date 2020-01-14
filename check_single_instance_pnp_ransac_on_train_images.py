from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler, pnp_ransac_multiple_instances, mode
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision.transforms import ToPILImage


def main(kaggle_dataset_dir_path, show=False, save=False):

    dataset = KaggleImageMaskDataset(kaggle_dataset_dir_path, is_train=True)
    downscaling=8
    models_handler = ModelsHandler(kaggle_dataset_dir_path)

    # useful if you dont have all the masks
    some_mask_paths = glob(os.path.join(
        kaggle_dataset_dir_path, 'train_targets/*.npy'
    ))[:40]
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

        background_id = 79
        most_common_class = mode(classification_mask[classification_mask != background_id]).mode.item()
        model_id = most_common_class
        print('most common class', most_common_class)


        ###

        result = models_handler.pnp_ransac_single_instance(image[..., 1], image[..., 2], image[..., 0] == model_id, model_id,
                                                           downscaling, min_inliers=50)
        success, ransac_rotation_matrix, ransac_translation_vector, inliers = result

        if not success:
            continue

        print('succes')

        image = np.zeros_like(image)
        image = models_handler.draw_model(image, model_id, ransac_translation_vector, ransac_rotation_matrix, downscaling)
        plt.imshow(image);
        plt.show()

        print(ransac_translation_vector, sep='\n')
        print(ransac_rotation_matrix, sep='\n')

        ###

        result = models_handler.pnp_ransac_single_instance(
            height_mask, angle_mask, classification_mask==most_common_class,
            most_common_class, downscaling=8
        )
        success, rot, trans, inliers = result
        print(trans)
        print(prediction_string)
        plt.imshow(models_handler.draw_model(
            np.zeros([3000, 3000, 3], dtype=np.uint8),
            most_common_class,
            trans, rot, 1
        ))

        if show:
            plt.show()

        break


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser(description="run to check whether dataset return what it should")
    arg_parser.add_argument('kaggle_dataset_dir_path')
    arg_parser.add_argument('--show', action='store_true', help='show masks visualization with matplotlib (blocks)')
    arg_parser.add_argument('--save', action='store_true', help='save produced visualizations in sanity_checks')

    args = arg_parser.parse_args()
    main(args.kaggle_dataset_dir_path, args.show, args.save)