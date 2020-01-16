from tqdm import tqdm
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler
import pandas as pd
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np

def main(path_to_submission_file, path_to_kaggle_dataset_dir, path_to_output_dir, n_images_to_handle):
    dataset = KaggleImageMaskDataset(path_to_kaggle_dataset_dir, is_train=False)
    models_handler = ModelsHandler(path_to_kaggle_dataset_dir)
    
    # prediction strings indexed by ImageId
    submission = pd.read_csv('submission_50_inliers.csv').set_index('ImageId').fillna('').PredictionString

    n_images_handled = 0 
    for image, image_id in tqdm(zip(dataset, dataset.get_IDs())):
        print(image_id)
        prediction_string = submission[image_id]
        if not prediction_string:
            continue
        img = image.cpu().numpy().transpose(1,2,0)
        img = 256*(img-img.min())/(img.max()-img.min()).astype(np.uint8)

        save_dir = os.path.join(path_to_output_dir, image_id)
        os.makedirs(save_dir, exist_ok=True)

        # save orig image
        plt.figure()
        plt.imshow(img.astype(np.uint8))
        plt.savefig(os.path.join(save_dir, 'orig.jpg'))
        plt.close()

        # draw instances overlay
        img = np.zeros_like(img, dtype=np.uint8)
        models_handler.draw_kaggle_models_from_prediction_string(img, prediction_string, downsampling=8)
        plt.figure()
        plt.imshow(img.astype(np.uint8))
        plt.savefig(os.path.join(save_dir, 'instances.jpg'))
        plt.close()
        
        n_images_handled += 1
        if n_images_handled >= n_images_to_handle:
            break


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('path_to_submission_file')
    arg_parser.add_argument('path_to_kaggle_dataset_dir')
    arg_parser.add_argument('path_to_output_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='do only 20 images')

    args = arg_parser.parse_args()

    n_images_to_handle = 20 if args.debug else 10000

    main(
        args.path_to_submission_file,
        args.path_to_kaggle_dataset_dir,
        args.path_to_output_dir,
        n_images_to_handle
    )