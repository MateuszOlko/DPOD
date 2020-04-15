import os

import numpy as np
import pandas as pd

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1


def create_split(dataset_path, output_path=None):
    np.random.seed(seed=12345)

    if output_path is None:
        output_path = dataset_path

    pictures_path = os.path.join(dataset_path, 'RGB-D', 'rgb_noseg')
    pictures_files = [f for f in os.listdir(pictures_path) if os.path.isfile(os.path.join(pictures_path, f))]
    pictures_files_permuted = list(np.random.permutation(np.array(pictures_files)))
    n = len(pictures_files_permuted)
    train_pictures = pictures_files_permuted[:int(n * TRAIN_SIZE)]
    validation_pictures = pictures_files_permuted[int(n * TRAIN_SIZE):int(n * (TRAIN_SIZE + VAL_SIZE))]
    test_pictures = pictures_files_permuted[int(n * TRAIN_SIZE + VAL_SIZE):]

    for csv_name, picture_names in zip(
            ['train_data_images_split.csv', 'validation_data_images_split.csv', 'test_data_images_split.csv'],
            [train_pictures, validation_pictures, test_pictures]):
        pd.DataFrame(np.array(picture_names), columns=['filenames']).to_csv(os.path.join(output_path, csv_name),
                                                                            index=False)