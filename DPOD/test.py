import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
import os
from utils import tensor1_to_jpg, tensor3_to_jpg


def main(path_to_model, path_to_kaggle_folder, save=False):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = KaggleImageMaskDataset(path_to_kaggle_folder, is_train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=1
    )

    # load correspondence block
    model1 = DPOD(image_size=(2710 // 8, 3384 // 8))
    model1.load_state_dict(torch.load(path_to_model))
    model1.to(device)

    # load pose block
    model2 = PoseBlock(path_to_kaggle_folder)

    # configure saving
    if save:
        save_dir = os.path.join(
            os.path.split(path_to_model)[0],
            'viz')
        os.makedirs(save_dir, exist_ok=True)
        print(save_dir)

    with torch.no_grad():
        n_image = 0
        for images in data_loader:
            images = images.to(device)
            classes, u_channel, v_channel = model1(images)
            if save:
                for i, c, u, v in zip(images, classes, u_channel, v_channel):
                    tensor3_to_jpg(i, f'{save_dir}/{n_image}/orginal.jpg')
                    tensor1_to_jpg(c.argmax(dim=0), f'{save_dir}/{n_image}/class.jpg')
                    tensor1_to_jpg(u.argmax(dim=0), f'{save_dir}/{n_image}/height.jpg')
                    tensor1_to_jpg(v.argmax(dim=0), f'{save_dir}/{n_image}/angle.jpg')
                    n_image += 1
                    
            result = model2(classes, u_channel, v_channel)
            print(result)
            exit(0)

if __name__ == "__main__":
    main('experiments/DPOD/Jan-15-15:20/final-model.pt', '../datasets/kaggle', True)
