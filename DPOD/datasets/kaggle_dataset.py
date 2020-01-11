import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np
import os

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
    
    def __init__(self, path, is_train=True, num_of_colors=256, num_of_models=79):
        self.is_train = is_train
        self.images_dir = os.path.join(path, "train_images" if is_train else "test_images")
        self.masks_dir = os.path.join(path, "train_targets" if is_train else "test_targets")
        data_csv = pd.read_csv(os.path.join(path, "train.csv"))
        self.images_ID = data_csv.ImageId
        self.predition_strings = data_csv.PredictionString
        self.num_of_colors = num_of_colors
        self.num_of_models = num_of_models
        self.im_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((3384//4, 2710//4)),
        ])
        self.target_transform = transforms.Resize((3384//4, 2710//4))
        # TODO make sure alll values in target are integers

    def __len__(self):
        return len(self.images_ID)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.images_ID[idx]+".jpg")
        image = Image.open(img_name)
        image.convert("RGB")
        # image = np.array(image)
        # image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
        # image = image.astype("float32")
        # image /= 256

        if !self.is_train:
            return image

        mask_name = os.path.join(self.masks_dir, self.images_ID[idx]+".npy")
        masks = np.load(mask_name)

        classification = np.zeros((self.num_of_models + 1, masks.shape[1], masks.shape[2]))
        for i in range(-1, self.num_of_models):
            if i == -1:
                classification[self.num_of_models][masks[0] == -1] = 1
            else:
                classification[i][masks[0] == i] = 1

        correspondence_u = np.zeros((self.num_of_colors, masks.shape[1], masks.shape[2]))
        correspondence_v = np.zeros((self.num_of_colors, masks.shape[1], masks.shape[2]))
        for i in range(self.num_of_colors):
            correspondence_u[i][masks[1] == i] = 1
            correspondence_v[i][masks[2] == i] = 1

        prediction_string = self.predition_strings[idx]

        image = self.im_transform(image)
        classification = self.target_transform(classification)
        correspondence_u = self.target_transform(correspondence_u)
        correspondence_v = self.target_transform(correspondence_v)

        return image, (classification, correspondence_u, correspondence_v), prediction_string
