from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
import os
from DPOD.utils import tensor1_to_jpg, tensor3_to_jpg, array3_to_jpg
import numpy as np
from time import time


def main(path_to_model, path_to_kaggle_folder, render_visualizations=False):
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
    save_dir = os.path.join(
        os.path.split(path_to_model)[0],
        'output')
    os.makedirs(save_dir, exist_ok=True)
    print('saving to', save_dir)

    with torch.no_grad():
        for n_image, images in enumerate(tqdm(data_loader)):

            os.makedirs(f'{save_dir}/{n_image}', exist_ok=True)

            if os.path.exists(f'{save_dir}/{n_image}/ransac_calculated.empty') and not render_visualizations:
                continue

            batch_of_images = images.to(device)
            batch_of_classes, batch_of_u_channel, batch_of_v_channel = model1(batch_of_images)
            batch_of_instances = model2(batch_of_classes, batch_of_u_channel, batch_of_v_channel)

            image = batch_of_images[0]
            classes = batch_of_classes[0]
            u_channel = batch_of_u_channel[0]
            v_channel = batch_of_v_channel[0]
            instances = batch_of_instances[0]

            for n_instance, instance in enumerate(instances):
                model_id, translation_vector, rotation_matrix = instance
                np.save(f'{save_dir}/{n_image}/ransac_{n_instance}_translation_vector.npy', translation_vector)
                np.save(f'{save_dir}/{n_image}/ransac_{n_instance}_rotation_matrix.npy', rotation_matrix)
            open(f'{save_dir}/{n_image}/ransac_calculated.empty', 'a').close()

            if render_visualizations:
                tensor3_to_jpg(image, f'{save_dir}/{n_image}/orginal.jpg')
                tensor1_to_jpg(classes.argmax(dim=0), f'{save_dir}/{n_image}/class.jpg')
                tensor1_to_jpg(u_channel.argmax(dim=0), f'{save_dir}/{n_image}/height.jpg')
                tensor1_to_jpg(v_channel.argmax(dim=0), f'{save_dir}/{n_image}/angle.jpg')
                _, h, w = image.shape
                image_to_draw_ransac_overlays_onto = \
                    np.zeros((h, w, 3), dtype=np.uint8)
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


if __name__ == "__main__":
    main('experiments/DPOD/Jan-15-15:20/final-model.pt', '../datasets/kaggle', False)
