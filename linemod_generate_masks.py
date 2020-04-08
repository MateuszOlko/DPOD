import numpy as np
import cv2
from glob import glob
import os
from functools import lru_cache
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from linemod_models_handler import ModelsHandler, read_position_file


def get_all_image_ids(linemod_dir_path):
    return sorted([
        os.path.split(path)[1][len('color_'):-len('.png')]
        for path in glob(f'{linemod_dir_path}/RGB-D/rgb_noseg/*.png')
    ])


def generate_masks(linemod_dir_path, models_dir_path, target_dir_path, debug=False, show=False, save=False):
    models_handler = ModelsHandler(models_dir_path)
    os.makedirs(target_dir_path, exist_ok=True)
    os.makedirs(target_dir_path+'_viz', exist_ok=True)

    def target(image_id):
        save_path = f'{target_dir_path}/{image_id}_masks.npy'
        if not debug and os.path.exists(save_path):
            return

        data = []
        for model_name in models_handler.model_names:
            pose_file_path = f'{linemod_dir_path}/poses/{model_name}/info_{image_id}.txt'
            if not os.path.exists(pose_file_path):
                continue
            position = read_position_file(pose_file_path)
            if position == None:
                # no object
                continue
            _, _, rotation_matrix, center, _ = position
            data.append((model_name, rotation_matrix, center))

        correspondence_mask = np.zeros((480, 640, 2))#.astype(int)
        class_mask = np.zeros((480, 640))#.astype(int)
        for model_name, rotation_matrix, center in sorted(data, key=lambda x: x[2][2]):
            correspondence_mask = models_handler.draw_color_mask(correspondence_mask, model_name, rotation_matrix, center)
            class_mask          = models_handler.draw_class_mask(class_mask,          model_name, rotation_matrix, center)

        if show:
            plt.imshow(correspondence_mask[..., 0]); plt.draw(); plt.pause(0.5)
            plt.imshow(correspondence_mask[..., 1], cmap='twilight_shifted'); plt.draw(); plt.pause(0.5)
            plt.imshow(class_mask); plt.draw(); plt.pause(0.5)

        if save:
            # save visualizations
            plt.imsave(f'{target_dir_path}_viz/{image_id}_u_mask.jpg', correspondence_mask[..., 0])
            plt.imsave(f'{target_dir_path}_viz/{image_id}_v_mask.jpg', correspondence_mask[..., 1], cmap='twilight_shifted')
            plt.imsave(f'{target_dir_path}_viz/{image_id}_class_masks.jpg', class_mask)

        mask = np.stack([correspondence_mask[..., 0], correspondence_mask[..., 1], class_mask])
        np.save(save_path, mask)

    ids_to_process = get_all_image_ids(linemod_dir_path)
    if debug:
        ids_to_process = ids_to_process[:10]

    for image_id in tqdm(ids_to_process):
        target(image_id)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--linemod_dir_path', default='/mnt/bigdisk/datasets/linemod', help='path to directory containing linemod dataset')
    argparser.add_argument('--models_dir_path',  default='models_small', help='path to directory containing 3D models')
    argparser.add_argument('--target_dir_path',  default='/mnt/bigdisk/datasets/linemod/masks', help='path to directory to save generated masks to')
    argparser.add_argument('--show', action='store_true', help='show generated images on the go')
    argparser.add_argument('--debug', '-d', action='store_true', help='process only 20 images')
    argparser.add_argument('--save', action='store_true', help='save humanreadable masks')
    

    args = argparser.parse_args()

    linemod_dir_path = args.linemod_dir_path
    models_dir_path  = args.models_dir_path
    target_dir_path  = args.target_dir_path
    debug = args.debug
    show = args.show
    save = args.save
    print(args)

    generate_masks(linemod_dir_path, models_dir_path, target_dir_path, debug=debug, show=show, save=save)
