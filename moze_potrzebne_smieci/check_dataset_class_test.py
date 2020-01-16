from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchvision.transforms import ToPILImage


def main(kaggle_dataset_dir_path, inferenced_test_data_path="test_output",  show=False, save=False):

    dataset = KaggleImageMaskDataset(kaggle_dataset_dir_path, is_train=False)
    models_handler = ModelsHandler(kaggle_dataset_dir_path)

    if save:
        output_dir = f"sanity_checks_test/{time.strftime('%Y.%m.%dT%H:%M')}"
        os.makedirs(output_dir, exist_ok=True)

    for i, id_ in enumerate(dataset.get_IDs()):
        # load
        image = dataset[i]

        # to numpy format
        image = image.numpy().transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        image = (256*image).astype(np.uint8)

        net_output = np.load(inferenced_test_data_path + "/" + id_ + ".npy")
        
        model_type_img, height_img, angle_img = net_output[0], net_output[1], net_output[2]

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].imshow(image)
        axs[0, 1].imshow(model_type_img)
        axs[1, 0].imshow(height_img)
        axs[1, 1].imshow(angle_img)
        plt.tight_layout()

        if save:
            plt.savefig(f'{output_dir}/check_test-{id_}.jpg')

        if show:
            plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    arg_parser = ArgumentParser(description="run to check whether dataset return what it should")
    arg_parser.add_argument('kaggle_dataset_dir_path')
    arg_parser.add_argument('inferenced_test_data_path')
    arg_parser.add_argument('--show', action='store_true', help='show masks visualization with matplotlib (blocks)')
    arg_parser.add_argument('--save', action='store_true', help='save produced visualizations in sanity_checks')

    args = arg_parser.parse_args()
    main(args.kaggle_dataset_dir_path, args.show, args.save)