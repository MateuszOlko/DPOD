import numpy as np
import cv2
from DPOD.models_handler import ModelsHandler, euler_to_Rot
import matplotlib.pyplot as plt

def pnp_ransac_single_instance(color_u, color_v, mask, model_id, downscaling, models_handler, min_inliers=500, ):
    # todo handle picture scaling
    """
    :param color_u:  (h,w) np.uint8 array
    :param color_v:  (h,w) np.uint8 array
    :param mask:     (h,w) bool array - pixels to consider
    :param model_id: model to fit
    :return:
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
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2])

    try:
        result = cv2.solvePnPRansac(points_observed, points_projected, models_handler.camera_matrix, None)
    except cv2.error:
        return False, np.zeros([3, 3]), np.zeros(3), np.zeros([0, 2])

    success, ransac_rotataton_rodrigues_vector, ransac_translation_vector, inliers = result
    ransac_rotataton_rodrigues_vector = ransac_rotataton_rodrigues_vector.flatten()
    ransac_rotation_matrix = cv2.Rodrigues(ransac_rotataton_rodrigues_vector)[0].T
    ransac_translation_vector = ransac_translation_vector.flatten()
    if success:
        inliers = inliers.flatten()
        if len(inliers) < min_inliers:
            success = False

        pixels_of_inliers = np.stack(pixels_to_consider).T[inliers]
        return success, ransac_rotation_matrix, ransac_translation_vector, pixels_of_inliers
    else:
        return success, ransac_rotation_matrix, ransac_translation_vector, np.zeros((0, 2))


# test if it works

models_handler = ModelsHandler('../data/kaggle')


# prepare data
if True:
    # z palca
    data = np.zeros((2710, 3384, 3), dtype=np.uint8)
    translation_vector = np.array([-3, -2, 15])
    rotation_matrix = euler_to_Rot(0, 0, 0.7)
    rotation_rodrigues_vector = cv2.Rodrigues(rotation_matrix)[0]
    model_id = 5
    data = models_handler.draw_model(data, model_id, translation_vector, rotation_matrix, 1)
    class_mask, height_mask, angle_mask = data[..., 0], data[..., 1], data[..., 2]
else:
    # todo load iamge
    pass

# visualize data
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
axs[0, 0].imshow(class_mask)
axs[0, 1].imshow(height_mask)
axs[1, 0].imshow(angle_mask)
plt.show()

result = pnp_ransac_single_instance(height_mask, angle_mask, class_mask==model_id, model_id, 1, models_handler)
success, ransac_translation_vector, ransac_translation_vector, inliers = result
print(result)

axs[1, 1].imshow(models_handler.draw_model(
    np.zeros_like(data), model_id,
))
plt.show()