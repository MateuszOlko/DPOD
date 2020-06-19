from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
import pickle
from DPOD.datasets.linemod.models_handler import ModelsHandler
import cv2
from pprint import pprint

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
        color_u[pixels_to_consider].flatten(),
        color_v[pixels_to_consider].flatten(),
    ]).T.astype(int)

    # print(observed_colors.shape)
    points_observed = models_handler.get_color_to_3dpoints_arrays(model_name)[
        observed_colors[:, 0], observed_colors[:, 1]]
    # print("old p obs", points_observed.shape)
    # print("obs u", observed_colors[:, 0].shape)
    # observed_ys = models_handler.get_color_to_y(model_name)[observed_colors[:, 0]]
    # observed_xzs = models_handler.get_color_to_xz(model_name)[observed_colors[:, 1]]
    # print("obs zx", observed_xzs.shape)
    # points_observed = np.concatenate([observed_xzs[:, 0], observed_ys, observed_xzs[:, 1]], axis=-1)
    points_projected = np.stack([pixels_to_consider[1], pixels_to_consider[0]]).T.astype(float)
    # print("pp", points_projected.shape)
    # print("po", points_observed.var(axis=0), points_observed.shape)
    if len(points_observed) < 6 or len(points_projected) < 6:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_name
    try:
        result = cv2.solvePnPRansac(points_observed, points_projected, models_handler.camera_matrix, None, **solvePnPRansacKwargs)
    except cv2.error as e:
        print("Error", e)
        print(len(points_observed), len(points_projected))
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_name

    success, ransac_rotataton_rodrigues_vector, ransac_translation_vector, inliers = result
    ransac_rotataton_rodrigues_vector = ransac_rotataton_rodrigues_vector.flatten()
    ransac_rotation_matrix = cv2.Rodrigues(ransac_rotataton_rodrigues_vector)[0].T
    ransac_translation_vector = ransac_translation_vector.flatten()
    if success:
        inliers = inliers.flatten()
        #if len(inliers) < min_inliers:
        #    success = False

        pixels_of_inliers = np.stack(pixels_to_consider).T[inliers]
        return success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers, model_name
    else:
        print("inliers:", len(inliers) if inliers else 0)
        return success, ransac_rotation_matrix, ransac_translation_vector, np.zeros((0, 2)), model_name


def threaded_main(mask_path, models_handler, path_to_output_dir, min_inliers, verbose, solvePnPRansacKwargs):
    if verbose:
        print('processing', mask_path)
    instances = []
    color_u, color_v, class_mask = np.load(mask_path)
    for model_id in np.unique(class_mask):
        if model_id == 0:
            # background
            continue
        model_name = models_handler.model_id_to_model_name[int(model_id)]
        result = pnp_ransac_single_instance(color_u, color_v, class_mask == model_id, model_name, models_handler,
                                            min_inliers=min_inliers, **solvePnPRansacKwargs)
        success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers, model_name = result
        if success:
            print(f"Recognized model {model_id}")
            instances.append([model_name, ransac_translation_vector, ransac_rotation_matrix])

    image_id = os.path.split(mask_path)[1][:-len('_masks.npy')]
    output_file_path = f'{path_to_output_dir}/{image_id}_instances.pkl'
    if verbose:
        pprint(instances)
    with open(output_file_path, 'wb') as file:
        pickle.dump(instances, file)

    return None


def main(path_to_masks_dir, path_to_output_dir, min_inliers=50, debug=False, verbose=False, **solvePnPRansacKwargs):
    
    models_handler = ModelsHandler()

    os.makedirs(path_to_output_dir, exist_ok=True)
    n_masks_to_process = 20 if debug else 100000

    masks_paths = sorted(glob(f'{path_to_masks_dir}/*.npy'))[:n_masks_to_process]

    t = tqdm(total=len(masks_paths))
    with ProcessPoolExecutor() as executor:
        for _ in executor.map(
            threaded_main,
            masks_paths,
            [models_handler] * len(masks_paths),
            [path_to_output_dir] * len(masks_paths),
            [min_inliers] * len(masks_paths),
            [verbose] * len(masks_paths),
            [solvePnPRansacKwargs] * len(masks_paths),
        ):
            t.update()


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="""
    Applies ransac with specified parameters to all (3,h,w) np.uint8 
    path_to_masks_dir/<ImageId>_masks.npy masks with u, v, class channels
    and saves
    [
        [
            model_name,                   str
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
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='print predicted poses')

    arg_parser.add_argument('--min_inliers', type=int, default=50, help='handcrafted RANSAC parameter')

    arg_parser.add_argument('--iterationsCount', type=int, help='RANSAC parameter')
    arg_parser.add_argument('--reprojectionError', type=float, help='RANSAC parameter')
    arg_parser.add_argument('--confidence', type=float, help='RANSAC parameter')
    arg_parser.add_argument('--flags', type=str, help='RANSAC parameter')

    args = arg_parser.parse_args()

    solvePnPRansacKwargs = dict()
    for key in ['iterationsCount', 'reprojectionError', 'confidence', 'flags']:
        val = getattr(args, key)
        if val:
            solvePnPRansacKwargs[key] = val

    # solvePnPRansacKwargs['flags'] = cv2.SOLVEPNP_DLS

    main(args.path_to_masks_dir, args.path_to_output_dir, debug=args.debug, verbose=args.verbose, **solvePnPRansacKwargs)


