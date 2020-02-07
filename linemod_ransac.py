from tqdm import tqdm
import torch
from DPOD.model import PoseBlock
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
import pickle
from DPOD.datasets import PATHS
import json
from linemod_models_handler import ModelsHandler
import cv2


def pnp_ransac_single_instance(color_u, color_v, mask, model_name, models_handler: ModelsHandler, min_inliers=50, **solvePnPRansacKwargs):    
    """
    :param color_u:         (h,w) np.uint8 array
    :param color_v:         (h,w) np.uint8 array
    :param mask:            (h,w) bool array - pixels to consider
    :param model_id:        model to fit
    :param models_handler   ModelsHandler
    :param min_inliers      minimum number of inliers in fitted model for it to be accepted as valid
    :return: tuple
        success                     bool
        ransac_rotation_matrix      ...
        ransac_translation_vector   ...
        pixels_of_inliers           (n,2) int array with coordinates of pixels classified as inliers
        model_name                  model_name
    """

    #points, _ = models_handler.model_id_to_vertices_and_triangles(model_id)
    points = models_handler.get_vertices(model_name)
    pixels_to_consider = np.where(mask)

    observed_colors = np.stack([
        color_u[pixels_to_consider],
        color_v[pixels_to_consider]
    ]).T.astype(int)

    points_observed = models_handler.get_color_to_3dpoints_arrays(model_name)[
        observed_colors[:, 0], observed_colors[:, 1]]
    points_projected = np.stack([pixels_to_consider[1], pixels_to_consider[0]]).T.astype(float)

    if len(points_observed) < 6:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_name
    try:
        result = cv2.solvePnPRansac(points_observed, points_projected, models_handler.camera_matrix, None, **solvePnPRansacKwargs)
    except cv2.error:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_name

    success, ransac_rotataton_rodrigues_vector, ransac_translation_vector, inliers = result
    ransac_rotataton_rodrigues_vector = ransac_rotataton_rodrigues_vector.flatten()
    ransac_rotation_matrix = cv2.Rodrigues(ransac_rotataton_rodrigues_vector)[0].T
    ransac_translation_vector = ransac_translation_vector.flatten()
    if success:
        inliers = inliers.flatten()
        if len(inliers) < min_inliers:
            success = False

        pixels_of_inliers = np.stack(pixels_to_consider).T[inliers]
        return success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers, model_name
    else:
        return success, ransac_rotation_matrix, ransac_translation_vector, np.zeros((0, 2)), model_name


def apply_ransac(path_to_masks_dir, ransac_block, path_to_output_dir, debug=False):
    ###########  do wywalenia  ###########
    # prepare output dir
    os.makedirs(path_to_output_dir, exist_ok=True)

    n_masks_to_process = 20 if debug else 100000

    skipped = 0
    with torch.no_grad():
        for n_mask, mask_path in enumerate(tqdm(masks_paths)):
            if n_mask >= n_masks_to_process:
                break

            image_id = os.path.split(mask_path)[1][:-4]
            output_file_path = os.path.join(
                path_to_output_dir,
                f'{image_id}_instances.pkl'
            )

            if os.path.exists(output_file_path) and not debug:
                skipped += 1
                continue

            tensor = torch.tensor(np.load(mask_path))
            class_, u_channel, v_channel = tensor
            ransac_instances = ransac_block(class_, u_channel, v_channel)

            with open(output_file_path, 'wb') as file:
                pickle.dump(ransac_instances, file)
    print(f"Skipped processing of {skipped} images - already processed")


def main(path_to_masks_dir, path_to_output_dir, min_inliers=50, debug=False, verbose=False, **solvePnPRansacKwargs):
    
    models_handler = ModelsHandler()

    os.makedirs(path_to_output_dir, exist_ok=True)
    n_masks_to_process = 20 if debug else 100000

    masks_paths = sorted(glob(f'{args.path_to_masks_dir}/*.npy'))[:n_masks_to_process]
    for mask_path in tqdm(masks_paths):
        if verbose:
            print('processing', masks_path)
        instances = []
        color_u, color_v, class_mask = np.load(mask_path)
        for model_id in np.unique(class_mask):
            if model_id == 0:
                # background
                continue
            model_name = models_handler.model_id_to_model_name[model_id]
            result = pnp_ransac_single_instance(color_u, color_v, class_mask == model_id, model_name, models_handler, min_inliers=min_inliers, **solvePnPRansacKwargs)
            success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers, model_name = result
            if success:
                instances.append([model_name, ransac_translation_vector, ransac_rotation_matrix])
        
        image_id = os.path.split(mask_path)[1][:-len('_masks.npy')]
        output_file_path = f'{path_to_output_dir}/{image_id}_instances.pkl'
        if verbose:
            print(instances)
        with open(output_file_path, 'wb') as file:
            pickle.dump(instances, file)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="""
    Applies ransac with specified parameters to all (3,h,w) np.uint8 
    path_to_masks_dir/<ImageId>_masks.npy masks with u, v, class channels
    and saves
    [
        [
            model_id,                     int
            ransac_translation_vector,    (3)   float np.array
            ransac_rotation_matrix,       (3,3) float np.array
        ]
        for each instance found
    ]
    to path_to_output_dir/<ImageId>_instances.pkl with pickle.dump
    skipping calculating already saved outputs
    """)
    arg_parser.add_argument('path_to_masks_dir')
    arg_parser.add_argument('path_to_output_dir')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='process only 20 images')
    arg_parser.add_argument('-v', '--verbose', action='store_true')

    arg_parser.add_argument('--min_inliers', type=int, default=50)

    arg_parser.add_argument('--iterationsCount', type=int)
    arg_parser.add_argument('--reprojectionError', type=float)
    arg_parser.add_argument('--confidence', type=float)
    arg_parser.add_argument('--flags', type=str)

    args = arg_parser.parse_args()

    solvePnPRansacKwargs = dict()
    for key in ['iterationsCount', 'reprojectionError', 'confidence', 'flags']:
        val = getattr(args, key)
        if val:
            solvePnPRansacKwargs[key] = val

    main(args.path_to_masks_dir, args.path_to_output_dir, debug=args.debug, verbose=args.verbose, **solvePnPRansacKwargs)


