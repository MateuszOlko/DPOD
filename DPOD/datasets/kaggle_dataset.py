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


class KaggleImageMaskDataset(Dataset):
    """
    This class prepares masks for training
    For classification for N classes + background(denoted in files as -1) it
    prepares N+1 images of binary classification where (N+1)th is background

    For correspondence maps it prepares num_of_colors binary maps.

    Therefore element of dataset looks as follows:
    (image[W, H], (classification[N+1, W, H], u_channel[num_of_colors, W, H], v_channel[num_of_colors, W, H]), prediction_string)

    where W, H = 3384//4, 2710//4
    """
    
    def __init__(self, path, is_train=True, num_of_colors=256, num_of_models=79, image_size=(3384//8, 2710//8, )):
        self.is_train = is_train
        self.image_size = image_size
        self.frequency_path = os.path.join(path, "frequency.npy")
        self.images_dir = os.path.join(path, "train_images" if is_train else "test_images")
        self.masks_dir = os.path.join(path, "train_targets" if is_train else "test_targets")
        data_csv = pd.read_csv(os.path.join(path, "train.csv"))
        self.images_ID = data_csv.ImageId
        self.predition_strings = data_csv.PredictionString
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

        if not self.is_train:
            return image

        mask_name = os.path.join(self.masks_dir, self.images_ID[idx]+".npy")
        masks = np.load(mask_name)
        unknown_model_mask = masks[..., 0] >= self.num_of_models
        masks[unknown_model_mask, 0] = self.num_of_models
        masks = cv2.resize(masks, dsize=self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        masks = torch.tensor(masks, dtype=torch.uint8)

        prediction_string = self.predition_strings[idx]

        image = self.im_transform(image)

        classification_mask = masks[..., 0]
        height_mask         = masks[..., 1]
        angle_mask          = masks[..., 2]
        
        return image, (classification_mask, height_mask, angle_mask), prediction_string

    def get_class_weights(self, force=False):
        if os.path.exists(self.frequency_path) and not force:
            print("Loading frequency file...")
            response = np.load(self.frequency_path)
        else:
            print("Calculating frequency")
            paths = [os.path.join(self.masks_dir, self.images_ID[idx]+".npy") for idx in range(len(self.images_ID))]
            response = np.zeros(self.num_of_models+1)

            t = tqdm(total=len(self.images_ID))     
            with ProcessPoolExecutor() as executor:
                for freq in executor.map(get_class_weights, paths, [self.num_of_models]*len(self.images_ID)):
                    response += freq
                    t.update()

            np.save(self.frequency_path, response)

        zeros = response == 0
        response[zeros] = 1
        response = 1 / response
        response[zeros] = 0
        return torch.FloatTensor(response)
        

def get_class_weights(path, num_of_models):
        masks = np.load(path)
        unknown_model_mask = masks[..., 0] >= num_of_models
        masks[unknown_model_mask, 0] = num_of_models
        frequency = np.zeros(num_of_models +1)
        for i in np.unique(masks[..., 0]):
            frequency[i] += (masks[..., 0] == i).sum()
        return frequency
        



