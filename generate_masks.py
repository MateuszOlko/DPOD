import numpy as np
import pandas as pd
import cv2
from glob import glob
import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from time import time
from DPOD.models_handler import ModelsHandler
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('kaggle_dataset_path', default='data/kaggle')
    arg_parser.add_argument('mask_folder_path', default='data/kaggle/train_target')
    arg_parser.add_argument('-f', '--force', action='store_true', help='force calculating masks again')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='calculate only 10 masks, for debugging')
    arg_parser.add_argument('-p', '--parallel', action='store_true', help='use all cores')
    arg_parser.add_argument('-s', '--show', action='store_true', help='show masks visualization (blocks)')

    args = arg_parser.parse_args()

    N_IMAGES = 10 if args.debug else 1e10
    handler = ModelsHandler(args.kaggle_dataset_path)
    train_csv = pd.read_csv(f'{args.kaggle_dataset_path}/train.csv')
    os.makedirs(args.mask_folder_path, exist_ok=True)

    def target(train_image_path):
        image_id = os.path.split(train_image_path)[1][:-4]
        if (not args.force) and os.path.exists(f'{args.mask_folder_path}/{image_id}.npy'):
            print(f'skipping {image_id}')
            return
        print(f'processing {image_id}')

        kaggle_string = train_csv[train_csv.ImageId == image_id].PredictionString.iloc[0]
        output = handler.make_mask_from_kaggle_string(kaggle_string)
        np.save(f'{args.mask_folder_path}/{image_id}', output)
        if args.show:
            img = cv2.imread(train_image_path, cv2.COLOR_BGR2RGB)[:, :, ::-1]
            imgs = handler.make_visualizations(img, output)
            for img in imgs:
                plt.imshow(img)
                plt.show()

    tic = time()
    if args.parallel:
        print('using all cores')
        with ProcessPoolExecutor() as executor:
            executor.map(target, glob(f'{args.kaggle_dataset_path}/train_images/*.jpg')[:N_IMAGES])

    else:
        print('using only one core')
        for train_image_path in glob(f'{args.kaggle_dataset_path}/train_images/*.jpg')[:N_IMAGES]:
            target(train_image_path)

    print(f'took: {time()-tic:.2f} seconds')
