from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from glob import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from DPOD.datasets.linemod.models_handler import ModelsHandler, read_position_file
from DPOD import datasets


def get_all_image_ids(linemod_dir_path):
    return sorted([
        os.path.split(path)[1][len('color_'):-len('.png')]
        for path in glob(f'{linemod_dir_path}/RGB-D/rgb_noseg/*.png')
    ])


def draw_mask(models_handler, models_info, image_shape=(480, 640)):
    correspondence_mask = np.ones((*image_shape, 2)) * (models_handler.color_resolution + 1)  # .astype(int)
    class_mask = np.zeros(image_shape)  # .astype(int)
    for model_name, rotation_matrix, center in sorted(models_info, key=lambda x: x[2][2]):
        correspondence_mask = models_handler.draw_color_mask(correspondence_mask, model_name, rotation_matrix, center)
        class_mask = models_handler.draw_class_mask(class_mask, model_name, rotation_matrix, center)
    return correspondence_mask, class_mask


def prepare_models_info(model_names, poses_dir, image_id):
    data = []
    for model_name in model_names:
        pose_file_path = f'{poses_dir}/{model_name}/info_{image_id}.txt'
        if not os.path.exists(pose_file_path):
            continue
        position = read_position_file(pose_file_path)
        if position is None:  # no object
            continue
        _, _, rotation_matrix, center, _ = position
        data.append((model_name, rotation_matrix, center))
    return data


def process_image(image_id, models_handler, linemod_dir_path, target_dir_path, debug, show, save, force):
    save_path = f'{target_dir_path}/{image_id}_masks.npy'
    if not debug and os.path.exists(save_path) and not force:
        return

    data = prepare_models_info(models_handler.model_names, f'{linemod_dir_path}/poses', image_id)
    correspondence_mask, class_mask = draw_mask(models_handler, data)

    if show:
        plt.imshow(correspondence_mask[..., 0]);
        plt.draw();
        plt.pause(0.5)
        plt.imshow(correspondence_mask[..., 1], cmap='twilight_shifted');
        plt.draw();
        plt.pause(0.5)
        plt.imshow(class_mask);
        plt.draw();
        plt.pause(0.5)

    if save:
        # save visualizations
        plt.imsave(f'{target_dir_path}_viz/{image_id}_u_mask.jpg', correspondence_mask[..., 0])
        plt.imsave(f'{target_dir_path}_viz/{image_id}_v_mask.jpg', correspondence_mask[..., 1], cmap='twilight_shifted')
        plt.imsave(f'{target_dir_path}_viz/{image_id}_class_masks.jpg', class_mask)

    if np.any(np.unique(class_mask) == 8):
        print("ERROR")
    mask = np.stack([correspondence_mask[..., 0], correspondence_mask[..., 1], class_mask])
    np.save(save_path, mask)


def generate_masks(linemod_dir_path, models_dir_path, target_dir_path, debug=False, show=False, save=False, force=False):
    models_handler = ModelsHandler(models_dir_path)
    os.makedirs(target_dir_path, exist_ok=True)
    os.makedirs(f'{target_dir_path}_viz', exist_ok=True)

    ids_to_process = get_all_image_ids(linemod_dir_path)
    if debug:
        ids_to_process = ids_to_process[:10]

    # for image_id in tqdm(ids_to_process):
    #     process_image(image_id, models_handler, linemod_dir_path, target_dir_path, debug, show, save, force)

    t = tqdm(total=len(ids_to_process), smoothing=0.05)
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(
            process_image,
            ids_to_process,
            [models_handler] * len(ids_to_process),
            [linemod_dir_path] * len(ids_to_process),
            [target_dir_path] * len(ids_to_process),
            [debug] * len(ids_to_process),
            [show] * len(ids_to_process),
            [save] * len(ids_to_process),
            [force] * len(ids_to_process),
        ):
            t.update()
