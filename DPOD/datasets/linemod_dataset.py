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
    
    (image[H, W], (classification[N+1, H, W], u_channel[num_of_colors, H, W], v_channel[num_of_colors, H, W]), prediction_string)
    """

    def __init__(self, path, is_train=True, num_of_colors=256, num_of_models=7, image_size=(640, 480), setup=None):
        """
        :param setup    one of ["train", "test", "val", "all"]
        """
        self.is_train = is_train
        self.image_size = image_size
        self.images_dir = os.path.join(path, "RGB-D", "rgb_noseg")
        self.masks_dir = os.path.join(path, "masks")
        self.frequency_path = os.path.join(path, "frequency.npy")
        
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
        
        if self.is_train and setup == "all":
            # trening i walidacja złączone
            train_data_csv = pd.read_csv(os.path.join(path, "train_data_images_split.csv"))
            val_data_csv = pd.read_csv(os.path.join(path, "validation_data_images_split.csv"))
            
            self.images_filenames = pd.concat([train_data_csv, val_data_csv]).filenames
        if self.is_train and setup == "train":
            # sam trening
            self.images_filenames = pd.read_csv(os.path.join(path, "train_data_images_split.csv")).filenames
        if self.is_train and setup == "val":
            # sama walidacja
            self.images_filenames = pd.read_csv(os.path.join(path, "validation_data_images_split.csv")).filenames
        if not self.is_train:
            # sam test
            self.images_filenames = pd.read_csv(os.path.join(path, "test_data_images_split.csv")).filenames

        self.num_of_colors = num_of_colors
        self.num_of_models = num_of_models
        
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.images_filenames = np.array(self.images_filenames)

    def __len__(self):
        return len(self.images_ID)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.images_filenames[idx])
        image = Image.open(img_name)

        image = self.im_transform(image)
        
#         if not self.is_train:
#             return image

        mask_name = os.path.join(self.masks_dir, self.images_filenames[idx][:-4][6:]+"_masks.npy")
        masks = np.load(mask_name)
        unknown_model_mask = masks[..., 0] >= self.num_of_models
        masks[unknown_model_mask, 0] = self.num_of_models
        #masks = cv2.resize(masks, dsize=self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        masks = torch.tensor(masks, dtype=torch.uint8)

        height_mask         = masks[0]
        angle_mask          = masks[1]
        classification_mask = masks[2]
        
        return image, (classification_mask, height_mask, angle_mask)

    def get_class_weights(self, force=False):
        if os.path.exists(self.frequency_path) and not force:
            print("Loading frequency file...")
            frequency = np.load(self.frequency_path)
        else:
            print("Calculating frequency")
            list_of_train_masks = np.array(pd.read_csv(os.path.join(path, "train_data_images_split.csv")).filenames)
            paths = [os.path.join(self.masks_dir, filename[:-4][6:] + "_masks.npy") for filename in list_of_train_masks]
            class_frequency = np.zeros(self.num_of_models+1)
            height_frequency = np.zeros(self.num_of_colors)
            angle_frequency = np.zeros(self.num_of_colors)

            t = tqdm(total=len(self.images_ID))     
            with ProcessPoolExecutor() as executor:
                for freq in executor.map(get_class_frequencies, paths, [self.num_of_models]*len(self.images_ID)):
                    class_frequency += freq[0]
                    height_frequency += freq[1]
                    angle_frequency += freq[2]
                    t.update()

            np.save(self.frequency_path, frequency)

        
        results = []
        for frequency in [class_frequency, height_frequency, angle_frequency]:
            zeros = frequency == 0
            frequency[zeros] = 1
            weights = 1 / frequency
            weights[zeros] = 0
            weights *= (self.num_of_models + 1) / weights.sum()
            results.append(weights)
            
        results = [torch.FloatTensor(r) for r in results]    
        return results

        
def get_class_frequencies(path, num_of_models, num_of_colors):
        masks = np.load(path)
        class_frequency = np.zeros(num_of_models +1)
        height_frequency = np.zeros(num_of_colors)
        angle_frequency = np.zeros(num_of_colors)
        for i in np.unique(masks[2]):
            class_frequency[i] += (masks[2] == i).sum()
            
        for i in range(num_of_colors):
            height_frequency[i] += (masks[0] == i).sum()
            angle_frequency[i] += (masks[1] == i).sum()
        return class_frequency, height_frequency, angle_frequency
        



