import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
import os
from utils import tensor1_to_jpg, tensor3_to_jpg, array3_to_jpg
import numpy as np
from time import time

def main(path_to_model, path_to_kaggle_folder, save=False):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = KaggleImageMaskDataset(path_to_kaggle_folder, is_train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
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

    tic = time()
    with torch.no_grad():
        n_image = 0
        for images in data_loader:
            images = images.to(device)
            classes, u_channel, v_channel = model1(images)
            results = model2(classes, u_channel, v_channel)
        
            for i, c, u, v, instances in zip(images, classes, u_channel, v_channel, results):
                if save:
                    tensor3_to_jpg(i, f'{save_dir}/{n_image}/orginal.jpg')
                    tensor1_to_jpg(c.argmax(dim=0), f'{save_dir}/{n_image}/class.jpg')
                    tensor1_to_jpg(u.argmax(dim=0), f'{save_dir}/{n_image}/height.jpg')
                    tensor1_to_jpg(v.argmax(dim=0), f'{save_dir}/{n_image}/angle.jpg')
                    _, h, w = i.shape
                    image_to_draw_ransac_overlays_onto = \
                        np.zeros( (h, w, 3), dtype=np.uint8)
                    print(len(instances), 'found')
                    for n_instance, instance in enumerate(instances):
                        model_id, translation_vector, rotation_matrix = instance
                        image_to_draw_ransac_overlays_onto = model2.models_handler.draw_model(
                            image_to_draw_ransac_overlays_onto,
                            model_id,
                            translation_vector,
                            rotation_matrix,
                            model2.downscaling
                        )
                        print((image_to_draw_ransac_overlays_onto != 0).sum())
                        array3_to_jpg(
                            image_to_draw_ransac_overlays_onto,
                            f'{save_dir}/{n_image}/ransac_{n_instance}.jpg'
                        )

                n_image += 1                  
            
            print(*results, sep='\n')
            if n_image >= 50:
                break
    
    toc = time()
    print(f'took {toc-tic:.2f} s ({(toc-tic)/n_image:.2f} per image)')

if __name__ == "__main__":
    main('experiments/DPOD/Jan-15-15:20/final-model.pt', '../datasets/kaggle', False)
