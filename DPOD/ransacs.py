import numpy as np
import cv2
from DPOD.models_handler import ModelsHandler, euler_to_Rot
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
import matplotlib.pyplot as plt
from scipy.stats import mode


def pnp_ransac_single_instance(color_u, color_v, mask, model_id, downscaling, models_handler, min_inliers=500, ):
    # todo handle picture scaling
    """
    :param color_u:         (h,w) np.uint8 array
    :param color_v:         (h,w) np.uint8 array
    :param mask:            (h,w) bool array - pixels to consider
    :param model_id:        model to fit
    :param downscaling      downscaling factor (with respect to original 2710x3384 resolution) of provided masks
    :param models_handler   ModelsHandler
    :param min_inliers      minimum number of inliers in fitted model for it to be accepted as valid - todo: adjust to downscaling maybe
    :return: tuple
        success                     bool
        model_id                    model_id
        ransac_rotation_matrix      use it along translation vector and downsampling in ModelsHandler.draw_model
                                    for drawing proper overlay
        ransac_translation_vector   ...
        pixels_of_inliers           (n,2) int array with coordinates of pixels classified as inliers
        model_id                    model_id
    """
    points, _ = models_handler.model_id_to_vertices_and_triangles(model_id)
    pixels_to_consider = np.where(mask)

    observed_colors = np.stack([
        color_u[pixels_to_consider],
        color_v[pixels_to_consider]
    ]).T

    points_observed = models_handler.get_color_to_3dpoints_arrays(model_id)[
        observed_colors[:, 0], observed_colors[:, 1]]
    points_projected = np.stack([pixels_to_consider[1], pixels_to_consider[0]]).T.astype(float) * downscaling

    if len(points_observed) < 6:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_id

    try:
        result = cv2.solvePnPRansac(points_observed, points_projected, models_handler.camera_matrix, None)
    except cv2.error:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2]), model_id

    success, ransac_rotataton_rodrigues_vector, ransac_translation_vector, inliers = result
    ransac_rotataton_rodrigues_vector = ransac_rotataton_rodrigues_vector.flatten()
    ransac_rotation_matrix = cv2.Rodrigues(ransac_rotataton_rodrigues_vector)[0].T
    ransac_translation_vector = ransac_translation_vector.flatten()
    if success:
        inliers = inliers.flatten()
        if len(inliers) < min_inliers:
            success = False

        pixels_of_inliers = np.stack(pixels_to_consider).T[inliers]
        return success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers, model_id
    else:
        return success, ransac_rotation_matrix, ransac_translation_vector, np.zeros((0, 2)), model_id


def pnp_ransac_multiple_instance(class_, color_u, color_v, downscaling, models_handler, num_of_classes, min_inliers=1000):
    # TODO: adaptive min inliers
    # TODO: ignororowanie także otoczki overlaya znalezoinego modelu
    """
    Algorithm is as follows
    1. Select most frequent class apart from background
    2. Perform pnp_ransac_single_instance on these pixels and this class
    3. Set pixels that would be under overlay of fitted models as background
    4. Iterate

    :param num_of_classes
    :param class_           (h, w) int array specyfying class per pixel with everything
                            outside {0, ..., num_of_classes-1} interpreted as background class
    Other params as in pnp_ransac_single_instance

    :return: list of outputs such as in pnp_ransac_single_instance
    """

    if not np.logical_and(class_ >= 0, class_ < num_of_models).any():
        return []
    model_id = mode(class_[np.logical_and(class_ >= 0, class_ < num_of_models)]).mode.item()
    result = pnp_ransac_single_instance(
        color_u, color_v, class_ == model_id, model_id,
        downscaling, models_handler, min_inliers=min_inliers
    )
    success, rot, trans, inliers, model_id = result
    if not success:
        return []
    else:
        print('found')
        color_not_to_be_colored = num_of_classes
        overlay = np.zeros(class_.shape + (3,), dtype=np.uint8)+color_not_to_be_colored
        overlay = models_handler.draw_model(
            overlay,
            model_id, trans, rot, downscaling
        )

        class_[overlay[..., 0] != color_not_to_be_colored] = num_of_classes  # todo: warto by dodać otoczkę jeszcze
        plt.imshow(class_); plt.show()
        return [result] + pnp_ransac_multiple_instance(
            class_, color_u, color_v,
            downscaling, models_handler, num_of_classes, min_inliers=min_inliers)


if __name__ == '__main__':
    # test if it works

    models_handler = ModelsHandler('../data/kaggle')

    ### SINGLE INSTANCE

    # prepare data
    z_palca = False
    if z_palca:
        # z palca
        data = np.zeros((2710, 3384, 3), dtype=np.uint8)+255
        translation_vectors = [
            np.array([-3, -2, 20]),
            np.array([3, -2, 12]),
            np.array([-3, -2, 8]),
        ]
        rotation_matrices = [
            euler_to_Rot(0, 0, 0.7),
            euler_to_Rot(1, 0, 0.7),
            euler_to_Rot(0, 2, 0.7),
        ]
        #rotation_rodrigues_vector = cv2.Rodrigues(rotation_matrix)[0]
        model_id = 5
        for trans, rot in zip(translation_vectors, rotation_matrices):
            data = models_handler.draw_model(data, model_id, trans, rot, 1)
        class_mask, height_mask, angle_mask = data[..., 0], data[..., 1], data[..., 2]

        downscaling = 8
        class_mask = cv2.resize(class_mask, tuple(x // downscaling for x in reversed(class_mask.shape)), cv2.INTER_NEAREST)
        height_mask = cv2.resize(height_mask, tuple(x // downscaling for x in reversed(height_mask.shape)),cv2.INTER_NEAREST)
        angle_mask = cv2.resize(angle_mask, tuple(x // downscaling for x in reversed(angle_mask.shape)), cv2.INTER_NEAREST)

    else:
        # z datasetu
        dataset = KaggleImageMaskDataset('../data/kaggle')
        img, (class_mask, height_mask, angle_mask), predstr = dataset[2562]
        class_mask = np.array(class_mask)
        height_mask = np.array(height_mask)
        angle_mask = np.array(angle_mask)
        model_id = mode(class_mask[np.logical_and(class_mask >=0, class_mask < dataset.num_of_models)]).mode.item()
        downscaling = 8
        print('sought class', model_id, 'provided prediction string', predstr)

    # visualize data
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    axs[0, 0].imshow(class_mask)
    axs[0, 1].imshow(height_mask)
    axs[1, 0].imshow(angle_mask)

    result = pnp_ransac_single_instance(height_mask, angle_mask, class_mask == model_id, model_id, downscaling, models_handler)
    success, rotation_matrix, translation_vector, inliers, model_id = result
    print(translation_vector)

    rendered_guess = np.zeros(class_mask.shape + (3,), dtype=np.uint8)
    rendered_guess = models_handler.draw_model(
        rendered_guess, model_id, translation_vector, rotation_matrix, downscaling)
    rendered_guess[inliers[:, 0], inliers[:, 1], 0] = 255  # marks inliers as red

    axs[1, 1].imshow(rendered_guess)
    plt.show()

    ### MULTIPLE INSTANCE

    num_of_models = 79
    result = pnp_ransac_multiple_instance(class_mask, height_mask, angle_mask, downscaling, models_handler, num_of_models, min_inliers=10)
    for success, rot, trans, inliers, model_id in result:
        print(trans)