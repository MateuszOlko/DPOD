import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np
import os
import cv2
from tqdm import tqdm
from time import time
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1

class LinemodDatasetSplitConfigurator():
    def __init__(self):
        pass
        
    def create_split(self, dataset_path, output_path=None):
        np.random.seed(seed=12345)
        
        if output_path is None:
            output_path = dataset_path
        
        pictures_path = os.path.join(dataset_path, 'RGB-D', 'rgb_noseg')
        pictures_files = [f for f in os.listdir(pictures_path) if os.path.isfile(os.path.join(pictures_path, f))]
        pictures_files_permuted = list(np.random.permutation(np.array(pictures_files)))
        n = len(pictures_files_permuted)
        train_pictures = pictures_files_permuted[:int(n*TRAIN_SIZE)]
        validation_pictures = pictures_files_permuted[int(n*TRAIN_SIZE):int(n*(TRAIN_SIZE+VAL_SIZE))]
        test_pictures = pictures_files_permuted[int(n*TRAIN_SIZE+VAL_SIZE):]
    
        for csv_name, picture_names in zip([ 'train_data_images_split.csv', 'validation_data_images_split.csv', 'test_data_images_split.csv'],[train_pictures, validation_pictures, test_pictures]):
            pd.DataFrame(np.array(picture_names), columns=['filenames']).to_csv(os.path.join(dataset_path, csv_name), index=False)

class LinemodImageMaskDataset(Dataset):
    """
    This class prepares masks for training
    For classification for N classes + background(denoted in files as -1) it
    prepares N+1 images of binary classification where (N+1)th is background

    For correspondence maps it prepares num_of_colors binary maps.

    Therefore element of dataset looks as follows:
    (image[H, W], (classification[N+1, H, W], u_channel[num_of_colors, H, W], v_channel[num_of_colors, H, W]), prediction_string)
    """

    def __init__(self, path, is_train=True, num_of_colors=256, num_of_models=79, image_size=(640, 480), setup=None):
        """
        :param setup    one of ["train", "test", "val", "all"]
        """
        self.is_train = is_train
        self.image_size = image_size
        #self.frequency_path = os.path.join(path, "frequency.npy")
        self.images_dir = os.path.join(path, "train_images" if is_train else "test_images")
        self.masks_dir = os.path.join(path, "train_targets" if is_train else "test_targets")
        # This ifs result from backward compatibility
        if setup is not None:
            if setup == "train":
                self.is_train = True
            elif setup == "test":
                self.is_train = False
            elif setup == "val":
                self.is_train = True
            elif setup == "all":
                self.is_train = True
            else:
                raise AttributeError(f"No such setup: {setup}")
        setup = "all" if setup is None else setup
            
        if self.is_train:
            data_csv = pd.read_csv(os.path.join(path, "train.csv"))
            self.images_ID = data_csv.ImageId
            self.prediction_strings = data_csv.PredictionString
            edge = int(len(self.images_ID) * self.VAL_SIZE)
            if setup == "train":
                self.images_ID = self.images_ID[edge:]
                self.prediction_strings = self.prediction_strings[edge:]
            elif setup == "val":
                self.images_ID = self.images_ID[:edge]
                self.prediction_strings = self.prediction_strings[:edge]
            self.prediction_strings = list(self.prediction_strings)
        else:
            self.images_ID = [x[:-4] for x in os.listdir(os.path.join(path, "test_images"))]
            self.prediction_strings = None
        self.images_ID = list(self.images_ID)

        self.num_of_colors = num_of_colors
        self.num_of_models = num_of_models
        self.im_transform = transforms.Compose([
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size, Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images_ID)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.images_ID[idx]+".jpg")
        image = Image.open(img_name)

        image = self.im_transform(image)
        
        if not self.is_train:
            return image

        mask_name = os.path.join(self.masks_dir, self.images_ID[idx]+".npy")
        masks = np.load(mask_name)
        unknown_model_mask = masks[..., 0] >= self.num_of_models
        masks[unknown_model_mask, 0] = self.num_of_models
        masks = cv2.resize(masks, dsize=self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        masks = torch.tensor(masks, dtype=torch.uint8)

        prediction_string = self.prediction_strings[idx]

        classification_mask = masks[..., 0]
        height_mask         = masks[..., 1]
        angle_mask          = masks[..., 2]
        
        return image, (classification_mask, height_mask, angle_mask), prediction_string

    def get_class_weights(self, force=False):
        if os.path.exists(self.frequency_path) and not force:
            print("Loading frequency file...")
            frequency = np.load(self.frequency_path)
        else:
            print("Calculating frequency")
            paths = [os.path.join(self.masks_dir, self.images_ID[idx]+".npy") for idx in range(len(self.images_ID))]
            frequency = np.zeros(self.num_of_models+1)

            t = tqdm(total=len(self.images_ID))     
            with ProcessPoolExecutor() as executor:
                for freq in executor.map(get_class_frequencies, paths, [self.num_of_models]*len(self.images_ID)):
                    frequency += freq
                    t.update()

            np.save(self.frequency_path, frequency)

        zeros = frequency == 0
        frequency[zeros] = 1
        weights = 1 / frequency
        weights[zeros] = 0
        weights *= (self.num_of_models + 1) / weights.sum()
        return torch.FloatTensor(weights)
    
    def get_IDs(self):
        return self.images_ID
        
def get_class_frequencies(path, num_of_models):
        masks = np.load(path)
        unknown_model_mask = masks[..., 0] >= num_of_models
        masks[unknown_model_mask, 0] = num_of_models
        frequency = np.zeros(num_of_models +1)
        for i in np.unique(masks[..., 0]):
            frequency[i] += (masks[..., 0] == i).sum()
        return frequency
        



