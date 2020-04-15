import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from PIL import Image


class LinemodPoseDataset(Dataset):
    def __init__(self, path, setup, num_of_color_channels=256, num_of_models=7, image_size=(480, 640)):
        """
        :param setup    one of ["train", "test", "val", "all"]
        """
        super().__init__()
        self.path = path
        self.image_size = image_size
        self.images_dir = os.path.join(path, "RGB-D", "rgb_noseg")

        self._assure_dataset_split()

        if setup == "all":
            train_data_csv = pd.read_csv(os.path.join(path, "train_data_images_split.csv"))
            val_data_csv = pd.read_csv(os.path.join(path, "validation_data_images_split.csv"))
            test_data_csv = pd.read_csv(os.path.join(path, "test_data_images_split.csv"))
            self.images_filenames = pd.concat([train_data_csv, val_data_csv, test_data_csv]).filenames
        elif setup == "train":
            self.images_filenames = pd.read_csv(os.path.join(path, "train_data_images_split.csv")).filenames
        elif setup == "val":
            self.images_filenames = pd.read_csv(os.path.join(path, "validation_data_images_split.csv")).filenames
        elif setup == "test":
            self.images_filenames = pd.read_csv(os.path.join(path, "test_data_images_split.csv")).filenames
        else:
            raise AttributeError(f"No such setup: {setup}")

        self.num_of_color_channels = num_of_color_channels
        self.num_of_models = num_of_models

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.images_filenames = np.array(self.images_filenames)

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.images_filenames[idx])
        image = Image.open(img_name)

        image = self.im_transform(image)

        #from linemod_generate_masks
        models_data = prepare_models_info(models_handler.model_names, f'{linemod_dir_path}/poses', image_id)

        return image, rendered_model, pose

    def _assure_dataset_split(self):
        for filename in ['train_data_images_split.csv', 'validation_data_images_split.csv',
                         'test_data_images_split.csv']:
            if not os.path.exists(os.path.join(self.path, filename)):
                print("Creating split...")
                LinemodDatasetSplitConfigurator.create_split(self.path)
                break